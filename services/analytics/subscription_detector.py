import json
import os
import numpy as np
from datetime import datetime, timedelta
import uuid
import re
import hashlib
from collections import Counter
from sqlmodel import select
from config import SessionLocal
from models.database_models import Transaction, Subscription
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from prefect import get_run_logger
import logging

def _get_logger():
    """Fallback to standard logger if not running inside a Prefect flow."""
    try:
        return get_run_logger()
    except Exception:
        return logging.getLogger(__name__)

class SubscriptionDetector:
    def __init__(self, time_variance_threshold_days=20.0, price_hike_threshold=1.05, similarity_threshold=0.80):
        self.time_variance_threshold_days = time_variance_threshold_days
        self.price_hike_threshold = price_hike_threshold
        self.similarity_threshold = similarity_threshold
        
        base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")
        if not base_url.endswith("/v1"): 
            base_url = f"{base_url}/v1"
            
        # Nomic model for Vector Generation
        self.embeddings = OpenAIEmbeddings(
            base_url=base_url,
            model="text-embedding-nomic-embed-text-v1.5",
            api_key="budai-local",
            check_embedding_ctx_length=False
        )
        
        # Qwen LLM for Entity Extraction/Naming
        self.chat_model = ChatOpenAI(
            base_url=base_url,
            model="lmstudio-community/Qwen3.5-9B-GGUF",
            api_key="budai-local",
            temperature=0
        )

    def _cosine_similarity(self, v1, v2):
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0: 
            return 0.0
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def _clean_structural_noise(self, text):
        if not text: return ""
        # Aggressive pre-processing to stop false-positive structural clustering
        t = re.sub(r'card transaction of.*?issued by', '', text, flags=re.IGNORECASE)
        t = re.sub(r'\b\d+(\.\d+)?\s*(usd|eur|gbp)\b', '', t, flags=re.IGNORECASE)
        t = re.sub(r'\(fee:.*?\)', '', t, flags=re.IGNORECASE)
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    def analyze_user_subscriptions(self, user_uuid: str):
        logger = _get_logger()
        logger.info(f"Starting Semantic Subscription Detection ETL for user {user_uuid}")
        
        with SessionLocal() as session:
            # Phase 1: Extract & Partition
            pools = self.extract_and_partition(session, user_uuid)
            if not pools:
                logger.info(f"No subscription candidates found for user {user_uuid}")
                return

            # Phase 2: RAG Intercept (Placeholder for Mixed Quantization DB lookup)
            # e.g., pools = self.fast_vector_intercept(session, pools)
            
            # Phase 3: Transform & Cluster
            clusters = self.generate_and_cluster(pools)
            
            # Phase 4: Load & Storage Quantization
            self.upsert_and_archive(session, user_uuid, clusters)
            
    def extract_and_partition(self, session, user_uuid):
        logger = _get_logger()
        # Strictly filter data to reduce O(N^2) load
        txs = session.execute(
            select(Transaction)
            .where(Transaction.user_uuid == user_uuid)
            .where(Transaction.amount < 0)
            .where(Transaction.category.in_([
                "Subscriptions & Digital Services", 
                "Utilities", 
                "Entertainment & Lifestyle",
                "Healthcare"
            ]))
            .order_by(Transaction.date.asc())
        ).scalars().all()
        
        anti_grocery_keywords = [
            "tesco", "sainsburys", "asda", "morrisons", "aldi", "lidl", 
            "waitrose", "iceland", "co-op", "deliveroo", "uber eats", 
            "just eat", "mcdonalds", "kfc", "dominos", "food"
        ]
        
        pools = {}
        for tx in txs:
            if not tx.description or not tx.date: continue
            raw_desc_lower = tx.description.lower()
            
            if any(k in raw_desc_lower for k in anti_grocery_keywords): continue
            if "sent money" in raw_desc_lower: continue
            
            # Macro-Pooling: Group strictly by account, ignoring non-deterministic category tags
            key = tx.account_id
            if key not in pools: 
                pools[key] = []
            pools[key].append(tx)
            
        logger.info(f"Extracted and partitioned {len(txs)} transactions into {len(pools)} semantic micro-pools.")
        return pools

    def generate_and_cluster(self, pools):
        logger = _get_logger()
        valid_clusters = []
        
        for account_id, txs in pools.items():
            if len(txs) < 3: continue
            
            # Normalization: Strip structural noise but keep random hashes
            descriptions = [self._clean_structural_noise(tx.semi_cleaned_description or tx.description).strip().lower() for tx in txs]
            
            try:
                # Ping Alienware Nomic Server for Embeddings
                vectors = self.embeddings.embed_documents(descriptions)
            except Exception as e:
                logger.error(f"Failed to generate embeddings for account {account_id}: {e}")
                continue
                
            # Centroid Cosine Clustering Algorithm
            unassigned = list(range(len(txs)))
            while unassigned:
                anchor_idx = unassigned.pop(0)
                anchor_vector = vectors[anchor_idx]
                cluster_txs = [txs[anchor_idx]]
                
                i = 0
                while i < len(unassigned):
                    candidate_idx = unassigned[i]
                    sim = self._cosine_similarity(anchor_vector, vectors[candidate_idx])
                    
                    if sim >= self.similarity_threshold:
                        cluster_txs.append(txs[candidate_idx])
                        unassigned.pop(i)
                    else:
                        i += 1
                        
                # Rigid Periodicity and Variance Math
                if len(cluster_txs) >= 3:
                    dates = [t.date for t in cluster_txs]
                    dates.sort()
                    amounts = [abs(t.amount) for t in cluster_txs]
                    
                    time_gaps = [(dates[j] - dates[j-1]).days for j in range(1, len(dates))]
                    median_gap = np.median(time_gaps)
                    
                    if median_gap >= 5: # Filter burst purchases
                        std_gap = np.std(time_gaps)
                        if std_gap <= self.time_variance_threshold_days:
                            valid_clusters.append({
                                "anchor_tx": cluster_txs[0], # The semantic centroid
                                "account_id": account_id,
                                "transactions": cluster_txs,
                                "time_gaps": time_gaps,
                                "amounts": amounts,
                                "dates": dates,
                                "median_gap": median_gap
                            })
                            
        logger.info(f"Generated {len(valid_clusters)} valid subscription clusters across all pools.")
        return valid_clusters

    def upsert_and_archive(self, session, user_uuid, clusters):
        logger = _get_logger()
        detected_subscriptions = []
        from models.status_codes import PipelineStatus
        
        for c in clusters:
            anchor_tx = c["anchor_tx"]
            account_id = c["account_id"]
            cluster_txs = c["transactions"]
            dates = c["dates"]
            amounts = c["amounts"]
            time_gaps = c["time_gaps"]
            median_gap = c["median_gap"]
            
            # Forecast Frequency
            freq = "Monthly"
            if 6 <= median_gap <= 8: freq = "Weekly"
            elif 12 <= median_gap <= 16: freq = "Bi-Weekly"
            elif 80 <= median_gap <= 100: freq = "Quarterly"
            elif 110 <= median_gap <= 130: freq = "Termly"
            elif 350 <= median_gap <= 380: freq = "Annually"
            
            last_date = dates[-1]
            next_date = last_date + timedelta(days=int(np.mean(time_gaps)))
            latest_amount = amounts[-1]
            historical_avg = np.mean(amounts[:-1])
            is_hike = latest_amount > (historical_avg * self.price_hike_threshold)
            
            # LLM-Driven Presentation Cleanup (Entity Extraction)
            # Pass up to 3 descriptions from the cluster to give the LLM context without blowing up tokens
            cluster_descs = [tx.description for tx in cluster_txs[:3]]
            prompt = f"Extract the core commercial merchant brand name from these transaction descriptions. Output ONLY the brand name, nothing else. No punctuation.\nStrings: {cluster_descs}"
            
            try:
                response = self.chat_model.invoke(prompt)
                pretty_merchant_name = response.content.strip().title()
                
                # Failsafe if LLM hallucinated a whole paragraph
                if len(pretty_merchant_name.split()) > 4:
                    raise ValueError("LLM returned too many words")
            except Exception as e:
                logger.error(f"Failed to extract name via LLM: {e}")
                # Fallback to the cleaned string
                fallback_name = self._clean_structural_noise(anchor_tx.semi_cleaned_description or anchor_tx.description)
                pretty_merchant_name = fallback_name.title().split()[0] if fallback_name else "Unknown"
            
            # 90-day Expiration Pruning
            now_utc = datetime.utcnow().replace(tzinfo=None)
            days_overdue = (now_utc - next_date.replace(tzinfo=None)).days
            
            status_val = "303-410" if days_overdue > 90 else PipelineStatus.SUBSCRIPTION_DETECTED.value
            if is_hike and days_overdue <= 90:
                status_val = PipelineStatus.SUBSCRIPTION_PRICE_HIKE.value
                
            # Deterministic Cluster Hash (Solution 1)
            cleaned_strings = [self._clean_structural_noise(tx.semi_cleaned_description or tx.description).strip().lower() for tx in cluster_txs]
            mode_string = Counter(cleaned_strings).most_common(1)[0][0] if cleaned_strings else "unknown"
            hash_input = f"{account_id}_{mode_string}"
            cluster_signature_hash = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
            
            # Upsert Logic
            existing_sub = session.query(Subscription).filter_by(
                cluster_signature_hash=cluster_signature_hash
            ).first()
            
            if not existing_sub:
                # Fallback for legacy rows that don't have a hash yet
                existing_sub = session.query(Subscription).filter_by(
                    user_uuid=user_uuid, 
                    merchant_name=pretty_merchant_name,
                    account_id=account_id
                ).first()
            
            if existing_sub:
                if existing_sub.status == "303-410" or days_overdue > 90:
                    status_val = "303-410"
                    
                # Idempotency Skip Check
                if (existing_sub.next_expected_date == next_date.replace(tzinfo=None) and
                    existing_sub.last_payment_amount == float(latest_amount) and
                    existing_sub.status == status_val):
                    pass
                else:
                    existing_sub.last_payment_date = last_date.replace(tzinfo=None)
                    existing_sub.last_payment_amount = float(latest_amount)
                    existing_sub.next_expected_date = next_date.replace(tzinfo=None)
                    existing_sub.is_price_hike = bool(is_hike)
                    existing_sub.status = status_val
                    existing_sub.merchant_name = pretty_merchant_name
                    existing_sub.cluster_signature_hash = cluster_signature_hash
                    existing_sub.last_updated = datetime.utcnow()
                    detected_subscriptions.append(existing_sub)
            else:
                new_sub = Subscription(
                    subscription_uuid=str(uuid.uuid4()),
                    user_uuid=user_uuid,
                    merchant_name=pretty_merchant_name,
                    account_id=account_id,
                    bank_uuid=getattr(anchor_tx, "bank_uuid", None),
                    expected_amount=float(latest_amount),
                    last_payment_date=last_date.replace(tzinfo=None),
                    last_payment_amount=float(latest_amount),
                    predicted_frequency=freq,
                    next_expected_date=next_date.replace(tzinfo=None),
                    is_price_hike=bool(is_hike),
                    status=status_val,
                    cluster_signature_hash=cluster_signature_hash
                )
                session.add(new_sub)
                detected_subscriptions.append(new_sub)
                
            # Tag Mutation on Raw Transactions
            for tx in cluster_txs:
                existing_tags = tx.tags or []
                if "#recurring" not in existing_tags:
                    existing_tags.append("#recurring")
                if is_hike and tx.transaction_uuid == cluster_txs[-1].transaction_uuid:
                    if "#price-hike" not in existing_tags:
                        existing_tags.append("#price-hike")
                tx.tags = existing_tags
                
        # Final Atomic Commit
        if detected_subscriptions:
            session.commit()
            sub_names = [sub.merchant_name for sub in detected_subscriptions]
            logger.info(f"Successfully upserted {len(detected_subscriptions)} subscriptions into PostgreSQL: {', '.join(sub_names)}")
        else:
            session.commit()
            logger.info(f"No new material subscription updates required.")

        # Note: Database Storage Quantization (halfvec/bit integration for MerchantKnowledge) 
        # is structurally reserved here for when pgvector is explicitly installed on the database.
