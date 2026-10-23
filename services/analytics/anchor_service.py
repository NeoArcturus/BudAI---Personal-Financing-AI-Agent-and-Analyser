import re
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy import text
from config import SessionLocal
from services.embedding_service import EmbeddingService
from services.qdrant_service import get_qdrant_service
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

SIMILARITY_THRESHOLD = 0.88

def clean_merchant_string(raw_desc: str) -> str:
    """
    Normalizes bank transaction descriptions into clean canonical merchant labels.
    Removes common transaction prefixes, fee notices, card numbers, and web domains.
    """
    if not raw_desc or not raw_desc.strip():
        return "Unknown Merchant"

    text_clean = raw_desc.strip()

    # Strip prefixes like "Card transaction of 18.99 GBP issued by "
    text_clean = re.sub(
        r"(?i)^Card transaction of\s+[\d\.,]+\s+[A-Za-z]{3}\s+issued by\s+",
        "",
        text_clean
    )
    # Strip "Sent money to " / "Received money from "
    text_clean = re.sub(r"(?i)^(Sent money to|Received money from)\s+", "", text_clean)
    # Strip "Wise Charges for:? "
    text_clean = re.sub(r"(?i)^Wise Charges for:?\s*", "Wise Charges ", text_clean)
    # Strip trailing fee notices: "(fee: 0.01 GBP)"
    text_clean = re.sub(r"(?i)\(fee:.*?\)", "", text_clean)
    # Strip references like "with reference from Papa"
    text_clean = re.sub(r"(?i)with reference.*", "", text_clean)
    # Strip web URLs like g.co/helppay#, aws.amazon.co
    text_clean = re.sub(r"(?i)[a-z0-9\.\-]+\.(co|com|org|net|io|me)(/[^\s]*)?", "", text_clean)
    # Strip card identifiers: "CARD-4274153354", "*G1ai101m"
    text_clean = re.sub(r"(?i)CARD-\d+", "", text_clean)
    text_clean = re.sub(r"\*[A-Za-z0-9]+", "", text_clean)
    # Strip trailing hashes or digits
    text_clean = re.sub(r"#\w*", "", text_clean)
    text_clean = re.sub(r"\s+", " ", text_clean).strip(" -#*,:.")

    if not text_clean:
        return raw_desc[:50].strip()

    # Title-case for clean representation
    return text_clean.title()

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Computes cosine similarity between two float vectors.
    """
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

class AnchorService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.qdrant_service = get_qdrant_service()

    def process_and_anchor_batch(
        self,
        transactions: List[Dict[str, Any]],
        session
    ) -> List[Dict[str, Any]]:
        """
        Takes a list of transaction dictionaries (with 'semi_cleaned_description' or 'description'),
        computes embeddings, matches existing Qdrant merchant anchors (>= 0.88), clusters unmatched
        transactions within the batch into new anchors, persists new anchors to PostgreSQL and Qdrant,
        and assigns 'merchant_knowledge_uuid' to every transaction.
        """
        if not transactions:
            return transactions

        # Step 1: Prepare text to embed
        descriptions = [
            tx.get("semi_cleaned_description") or tx.get("description") or "Unknown"
            for tx in transactions
        ]

        # Step 2: Batch compute 768-dim embeddings in-process
        vectors = self.embedding_service.embed_documents(descriptions)

        # Step 3: Cluster incoming transactions in-memory first to minimize WAN latency
        # Clusters map representative index -> list of transaction indices belonging to it
        clusters: List[Dict[str, Any]] = []
        # Each cluster: {"rep_idx": int, "vector": List[float], "indices": List[int], "clean_name": str}

        for idx, tx in enumerate(transactions):
            v = vectors[idx]
            raw_desc = descriptions[idx]
            clean_name = clean_merchant_string(raw_desc)

            # Check if this vector matches an existing cluster in this batch
            matched_cluster = None
            best_sim = 0.0
            for cl in clusters:
                sim = cosine_similarity(v, cl["vector"])
                if sim >= SIMILARITY_THRESHOLD and sim > best_sim:
                    best_sim = sim
                    matched_cluster = cl

            if matched_cluster:
                matched_cluster["indices"].append(idx)
            else:
                clusters.append({
                    "rep_idx": idx,
                    "vector": v,
                    "indices": [idx],
                    "clean_name": clean_name
                })

        # Step 4: Resolve each cluster against Qdrant in batch or individually
        new_anchors: List[Dict[str, Any]] = []

        for cl in clusters:
            rep_vec = cl["vector"]
            clean_name = cl["clean_name"]

            # Query Qdrant for existing anchor
            hits = self.qdrant_service.search_merchant_knowledge(
                rep_vec,
                limit=1,
                score_threshold=SIMILARITY_THRESHOLD
            )

            if hits:
                matched_uuid = hits[0]["payload"].get("knowledge_uuid") or hits[0]["id"]
                for tx_idx in cl["indices"]:
                    transactions[tx_idx]["merchant_knowledge_uuid"] = matched_uuid
            else:
                # No existing anchor in Qdrant; create new anchor for this entire cluster
                new_uuid = str(uuid.uuid4())
                new_anchor = {
                    "knowledge_uuid": new_uuid,
                    "clean_merchant_name": clean_name,
                    "category": "Uncategorized",
                    "sub_category": None,
                    "tags": [],
                    "vector": rep_vec,
                    "is_human_verified": False
                }
                new_anchors.append(new_anchor)
                for tx_idx in cl["indices"]:
                    transactions[tx_idx]["merchant_knowledge_uuid"] = new_uuid

        # Step 5: Bulk persist newly created anchors into PostgreSQL
        if new_anchors:
            now = datetime.utcnow()
            for anchor in new_anchors:
                insert_stmt = text("""
                    INSERT INTO merchant_knowledge (
                        knowledge_uuid, clean_merchant_name, category, sub_category, tags, is_human_verified, created_at
                    ) VALUES (
                        :uuid, :name, :category, :sub_cat, '[]'::json, :verified, :now
                    ) ON CONFLICT (knowledge_uuid) DO NOTHING;
                """)
                session.execute(insert_stmt, {
                    "uuid": anchor["knowledge_uuid"],
                    "name": anchor["clean_merchant_name"],
                    "category": anchor["category"],
                    "sub_cat": anchor["sub_category"],
                    "verified": anchor["is_human_verified"],
                    "now": now
                })

            # Step 6: Batch upsert new anchor vectors into Qdrant
            try:
                self.qdrant_service.upsert_merchants_batch(new_anchors)
            except Exception as e:
                logger.error(f"Failed to upsert new merchant anchors into Qdrant: {e}")

        return transactions

    def reconcile_unlinked_transactions(self, limit: int = 500) -> Dict[str, Any]:
        """
        Scans PostgreSQL for transactions where merchant_knowledge_uuid IS NULL,
        vectorizes them, clusters them into Merchant Anchors, and updates both databases.
        """
        with SessionLocal() as session:
            query = text("""
                SELECT transaction_uuid, semi_cleaned_description, description, amount
                FROM transactions
                WHERE merchant_knowledge_uuid IS NULL
                LIMIT :limit;
            """)
            rows = session.execute(query, {"limit": limit}).fetchall()

            if not rows:
                return {
                    "status": "noop",
                    "unlinked_count": 0,
                    "anchors_created": 0,
                    "transactions_linked": 0
                }

            tx_dicts = [
                {
                    "transaction_uuid": r[0],
                    "semi_cleaned_description": r[1] or r[2] or "Unknown",
                    "description": r[2] or "Unknown",
                    "amount": float(r[3]) if r[3] else 0.0
                }
                for r in rows
            ]

            processed_txs = self.process_and_anchor_batch(tx_dicts, session)

            # Bulk update transactions in database
            for tx in processed_txs:
                if tx.get("merchant_knowledge_uuid"):
                    session.execute(
                        text("""
                            UPDATE transactions
                            SET merchant_knowledge_uuid = :mk_uuid
                            WHERE transaction_uuid = :tx_uuid;
                        """),
                        {
                            "mk_uuid": tx["merchant_knowledge_uuid"],
                            "tx_uuid": tx["transaction_uuid"]
                        }
                    )

            session.commit()

            unique_anchors = len(set(tx["merchant_knowledge_uuid"] for tx in processed_txs if tx.get("merchant_knowledge_uuid")))

            logger.info(
                f"Reconciliation complete: {len(processed_txs)} transactions processed, "
                f"{unique_anchors} distinct merchant anchors assigned."
            )

            return {
                "status": "success",
                "unlinked_count": len(rows),
                "anchors_created": unique_anchors,
                "transactions_linked": len(processed_txs)
            }

_anchor_service_instance: Optional[AnchorService] = None

def get_anchor_service() -> AnchorService:
    global _anchor_service_instance
    if _anchor_service_instance is None:
        _anchor_service_instance = AnchorService()
    return _anchor_service_instance
