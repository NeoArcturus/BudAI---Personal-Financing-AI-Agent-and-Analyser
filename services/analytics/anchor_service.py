import re
import uuid
import json
import time
import concurrent.futures
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy import text
from config import SessionLocal
from services.embedding_service import get_embedding_service
from services.qdrant_service import get_qdrant_service
from services.logger_setup import get_core_logger, log_pipeline_data

logger = get_core_logger(__name__)

SIMILARITY_THRESHOLD = 0.88

def clean_merchant_string(raw_desc: str) -> str:
    """
    Normalizes bank transaction descriptions into clean canonical merchant labels.
    Removes common transaction prefixes, fee notices, card numbers, currency codes,
    leading prices, and web domains.
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
    # Strip references like "with reference from Papa", "with reference Splitwise 144"
    text_clean = re.sub(r"(?i)with reference.*", "", text_clean)
    # Strip leading amount + currency patterns (e.g. "14.90 EUR ", "293.43 ", "2,070.00 ", "439.22 INR ")
    text_clean = re.sub(r"(?i)^(?:\d{1,3}(?:,\d{3})*|\d+)[\.,]\d{2}\s*(?:GBP|EUR|USD|INR)?\s*", "", text_clean)
    text_clean = re.sub(r"(?i)^(?:\d{1,3}(?:,\d{3})*|\d+)\s*(?:GBP|EUR|USD|INR)\s*", "", text_clean)
    # Strip leading currencies if left over
    text_clean = re.sub(r"(?i)^(?:EUR|GBP|USD|INR)\s+", "", text_clean)
    # Strip phone numbers (e.g. "0161 244 1000")
    text_clean = re.sub(r"\b0\d{2,4}\s?\d{3,4}\s?\d{3,4}\b", "", text_clean)
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
        self.embedding_service = get_embedding_service()
        self.qdrant_service = get_qdrant_service()

    def process_and_anchor_batch(
        self,
        transactions: List[Dict[str, Any]],
        session
    ) -> List[Dict[str, Any]]:
        """
        Takes a list of transaction dictionaries (with 'semi_cleaned_description' or 'description'),
        computes embeddings in parallel via FastEmbed ONNX, matches existing Qdrant merchant anchors,
        clusters unmatched transactions within the batch into new anchors, resolves LLM and vector search
        concurrently, persists new anchors to PostgreSQL and Qdrant in batch, and assigns 'merchant_knowledge_uuid'.
        """
        if not transactions:
            return transactions

        t_start_total = time.time()
        log_pipeline_data(
            logger=logger,
            pipeline="anchor_service",
            stage="batch_intake",
            data={
                "transaction_count": len(transactions),
                "sample": [
                    {
                        "desc": (tx.get("description") or tx.get("semi_cleaned_description") or "")[:40],
                        "amount": tx.get("amount")
                    }
                    for tx in transactions[:3]
                ]
            },
            message=f"Starting parallel anchor resolution and vectorisation for {len(transactions)} transactions"
        )

        # Step 1: Prepare text to embed (prioritizing full raw description)
        descriptions = [
            tx.get("description") or tx.get("semi_cleaned_description") or "Unknown"
            for tx in transactions
        ]

        # Step 2: Batch compute 384-dim embeddings in parallel via FastEmbed ONNX
        t0_vec = time.time()
        vectors = self.embedding_service.embed_documents(descriptions)
        vec_ms = int((time.time() - t0_vec) * 1000)

        # Step 3: Cluster incoming transactions in-memory first to minimize WAN latency
        t0_cluster = time.time()
        clusters: List[Dict[str, Any]] = []

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

        cluster_ms = int((time.time() - t0_cluster) * 1000)
        log_pipeline_data(
            logger=logger,
            pipeline="anchor_service",
            stage="clustering_completed",
            data={
                "transactions_total": len(transactions),
                "clusters_formed": len(clusters),
                "elapsed_ms": cluster_ms,
                "cluster_samples": [
                    {"rep": cl["clean_name"], "size": len(cl["indices"])}
                    for cl in clusters[:5]
                ]
            },
            message=f"Clustered {len(transactions)} transactions into {len(clusters)} distinct merchant entities in {cluster_ms}ms"
        )

        # Step 4: Resolve each cluster against LLM and Qdrant concurrently in parallel
        t0_resolve = time.time()

        def resolve_single_cluster(cl: Dict[str, Any]) -> Dict[str, Any]:
            rep_vec = cl["vector"]
            clean_name = cl["clean_name"]
            rep_tx = transactions[cl["rep_idx"]]
            rep_raw = descriptions[cl["rep_idx"]]
            rep_amt = float(rep_tx.get("amount") or 0.0)

            # Concurrent LLM Call + Qdrant Vector Search
            def _call_llm():
                try:
                    from agents.finance.Categorizer_Agent.CategorizerAgent import classify_transaction_with_llm
                    return classify_transaction_with_llm(
                        merchant_name=clean_name,
                        description=rep_raw,
                        amount=rep_amt
                    )
                except Exception as llm_err:
                    logger.warning(f"Primary LLM categorization for '{clean_name}' failed: {llm_err}")
                    return None

            def _call_qdrant():
                try:
                    return self.qdrant_service.search_merchant_knowledge(
                        rep_vec,
                        limit=1,
                        score_threshold=SIMILARITY_THRESHOLD
                    )
                except Exception as q_err:
                    logger.warning(f"Qdrant vector search for '{clean_name}' failed: {q_err}")
                    return []

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as inner_pool:
                f_llm = inner_pool.submit(_call_llm)
                f_qdrant = inner_pool.submit(_call_qdrant)
                llm_res = f_llm.result()
                hits = f_qdrant.result()

            llm_cat = None
            llm_sub = None
            llm_tags = []
            if llm_res and llm_res.get("category") and llm_res["category"] != "Uncategorized":
                llm_cat = llm_res["category"]
                llm_sub = llm_res.get("sub_category")
                llm_tags = llm_res.get("tags") or []

            source = "LLM" if llm_cat else ("QDRANT_CACHE" if hits else "REGEX_FALLBACK")

            if hits:
                payload = hits[0].get("payload", {})
                matched_uuid = payload.get("knowledge_uuid") or hits[0]["id"]
                matched_name = payload.get("clean_merchant_name") or clean_name
                matched_cat = llm_cat or payload.get("category") or "Uncategorized"
                matched_sub = llm_sub or payload.get("sub_category")
                matched_tags = llm_tags or payload.get("tags") or []
                matched_verified = bool(payload.get("is_human_verified", False))

                return {
                    "is_new": False,
                    "matched_uuid": matched_uuid,
                    "clean_name": matched_name,
                    "category": matched_cat,
                    "sub_category": matched_sub,
                    "tags": matched_tags,
                    "is_verified": matched_verified,
                    "indices": cl["indices"],
                    "source": source
                }
            else:
                # Novel anchor: needs creation and context vectorisation
                new_uuid = str(uuid.uuid4())
                init_cat = llm_cat
                init_sub = llm_sub
                if not init_cat or init_cat == "Uncategorized":
                    from services.categorization.classifier import classify_merchant
                    cat_rule, sub_rule = classify_merchant(clean_name, rep_raw, rep_amt)
                    init_cat = cat_rule if cat_rule != "Uncategorized" else "Shopping & Retail"
                    init_sub = sub_rule or "General Merchandise"

                from services.categorization.taxonomy import enrich_transaction_taxonomy
                matched_cat, matched_sub, enriched_tags, _ = enrich_transaction_taxonomy(
                    category=init_cat,
                    sub_category=init_sub,
                    description=f"{clean_name} {rep_raw}",
                    existing_tags=llm_tags
                )

                rich_context_str = f"{clean_name} | {matched_cat} | {matched_sub} | {', '.join(enriched_tags)}"
                return {
                    "is_new": True,
                    "knowledge_uuid": new_uuid,
                    "clean_merchant_name": clean_name,
                    "category": matched_cat,
                    "sub_category": matched_sub,
                    "tags": enriched_tags,
                    "rich_context_str": rich_context_str,
                    "fallback_vector": rep_vec,
                    "indices": cl["indices"],
                    "source": source
                }

        max_workers = min(8, max(1, len(clusters)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as cluster_pool:
            resolved_clusters = list(cluster_pool.map(resolve_single_cluster, clusters))

        resolve_ms = int((time.time() - t0_resolve) * 1000)
        llm_count = sum(1 for r in resolved_clusters if r["source"] == "LLM")
        qdrant_count = sum(1 for r in resolved_clusters if r["source"] == "QDRANT_CACHE")
        regex_count = sum(1 for r in resolved_clusters if r["source"] == "REGEX_FALLBACK")

        log_pipeline_data(
            logger=logger,
            pipeline="anchor_service",
            stage="parallel_resolution_completed",
            data={
                "clusters_count": len(clusters),
                "parallel_workers": max_workers,
                "llm_classified": llm_count,
                "qdrant_cache_hits": qdrant_count,
                "regex_fallbacks": regex_count,
                "elapsed_ms": resolve_ms
            },
            message=f"Parallel cluster resolution completed for {len(clusters)} clusters in {resolve_ms}ms (LLM={llm_count}, Qdrant={qdrant_count}, Regex={regex_count})"
        )

        # Step 5: Process existing anchors and batch-vectorise new anchors in parallel
        new_anchor_items = [r for r in resolved_clusters if r["is_new"]]
        new_anchors: List[Dict[str, Any]] = []

        if new_anchor_items:
            t0_anchor_vec = time.time()
            rich_context_strings = [item["rich_context_str"] for item in new_anchor_items]
            rich_vectors = self.embedding_service.embed_documents(rich_context_strings)
            anchor_vec_ms = int((time.time() - t0_anchor_vec) * 1000)

            log_pipeline_data(
                logger=logger,
                pipeline="anchor_service",
                stage="new_anchors_vectorised",
                data={
                    "new_anchors_count": len(new_anchor_items),
                    "elapsed_ms": anchor_vec_ms,
                    "sample": [
                        {"name": item["clean_merchant_name"], "cat": item["category"], "sub": item["sub_category"]}
                        for item in new_anchor_items[:3]
                    ]
                },
                message=f"Vectorised {len(new_anchor_items)} new merchant anchors in {anchor_vec_ms}ms in parallel"
            )

            for item, rich_vec in zip(new_anchor_items, rich_vectors):
                vec = rich_vec if rich_vec and any(rich_vec) else item["fallback_vector"]
                new_anchors.append({
                    "knowledge_uuid": item["knowledge_uuid"],
                    "clean_merchant_name": item["clean_merchant_name"],
                    "category": item["category"],
                    "sub_category": item["sub_category"],
                    "tags": item["tags"],
                    "vector": vec,
                    "is_human_verified": False
                })

        # Apply assignments to transactions and ensure matched anchors exist in PostgreSQL
        for r in resolved_clusters:
            if not r["is_new"]:
                try:
                    insert_hit_stmt = text("""
                        INSERT INTO merchant_knowledge (
                            knowledge_uuid, clean_merchant_name, category, sub_category, tags, is_human_verified, created_at
                        ) VALUES (
                            :uuid, :name, :category, :sub_cat, CAST(:tags AS json), :verified, :now
                        ) ON CONFLICT (knowledge_uuid) DO NOTHING;
                    """)
                    session.execute(insert_hit_stmt, {
                        "uuid": r["matched_uuid"],
                        "name": r["clean_name"],
                        "category": r["category"],
                        "sub_cat": r["sub_category"],
                        "tags": json.dumps(r["tags"]),
                        "verified": r["is_verified"],
                        "now": datetime.utcnow()
                    })
                except Exception as pg_err:
                    logger.warning(f"Failed ensuring matched anchor {r['matched_uuid']} in PostgreSQL: {pg_err}")

                for tx_idx in r["indices"]:
                    transactions[tx_idx]["merchant_knowledge_uuid"] = r["matched_uuid"]
                    if r["sub_category"]:
                        transactions[tx_idx]["sub_category"] = r["sub_category"]
                    if r["tags"]:
                        transactions[tx_idx]["tags"] = r["tags"]
            else:
                for tx_idx in r["indices"]:
                    transactions[tx_idx]["merchant_knowledge_uuid"] = r["knowledge_uuid"]
                    transactions[tx_idx]["sub_category"] = r["sub_category"]
                    transactions[tx_idx]["tags"] = r["tags"]

        # Step 6: Bulk persist newly created anchors into PostgreSQL
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

            # Step 7: Batch upsert new anchor vectors into Qdrant
            try:
                self.qdrant_service.upsert_merchants_batch(new_anchors)
            except Exception as e:
                logger.error(f"Failed to upsert new merchant anchors into Qdrant: {e}")

        # Step 8: Finalized summary logging and memory release
        total_ms = int((time.time() - t_start_total) * 1000)
        log_pipeline_data(
            logger=logger,
            pipeline="anchor_service",
            stage="batch_finalized",
            data={
                "total_transactions": len(transactions),
                "total_clusters": len(clusters),
                "new_anchors_created": len(new_anchors),
                "matched_anchors": len(clusters) - len(new_anchors),
                "total_elapsed_ms": total_ms
            },
            message=f"Anchor batch processing finalized in {total_ms}ms ({len(new_anchors)} new anchors, {len(clusters) - len(new_anchors)} matched)"
        )

        import gc
        del vectors
        del clusters
        gc.collect()

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
