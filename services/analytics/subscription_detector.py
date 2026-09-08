from langfuse.langchain import CallbackHandler
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
            model="Qwen3.5-9B-GGUF",
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

            # Phase 2: RAG Intercept & Vector Observability
            rag_clusters, unmatched_pools = self.fast_vector_intercept(session, pools)
            
            # Phase 3: Transform & Cluster (Only on Unmatched)
            llm_clusters = self.generate_and_cluster(unmatched_pools)
            
            # Combine Clusters
            all_clusters = rag_clusters + llm_clusters
            
            # Phase 4: Load & Storage Quantization
            if all_clusters:
                self.upsert_and_archive(session, user_uuid, all_clusters)

    def extract_and_partition(self, session, user_uuid):
        logger = _get_logger()
        # Strictly filter data to reduce O(N^2) load
        from models.database_models import MerchantKnowledge
        txs = session.execute(
            select(Transaction)
            .join(MerchantKnowledge, Transaction.merchant_knowledge_uuid == MerchantKnowledge.knowledge_uuid, isouter=True)
            .where(Transaction.user_uuid == user_uuid)
            .where(Transaction.amount < 0)
            .where(MerchantKnowledge.category.in_([
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


    def fast_vector_intercept(self, session, pools):
        from sqlalchemy import text
        import numpy as np
        
        logger = _get_logger()
        logger.info("Executing Phase 2: RAG Vector Intercept with Observability Telemetry")
        
        unmatched_pools = {}
        matched_groups = {} 
        
        os.makedirs("logs", exist_ok=True)
        log_file_path = "logs/vector_similarity.jsonl"
        
        for account_id, txs in pools.items():
            unmatched_pools[account_id] = []
            matched_groups[account_id] = {}
                
            for tx in txs:
                desc = tx.description or ""
                try:
                    query_vector = self.embeddings.embed_query(desc)
                except Exception as e:
                    logger.error(f"Embedding failed for tx {tx.transaction_uuid}: {e}")
                    unmatched_pools[account_id].append(tx)
                    continue
                
                # RAG Query with exact < 0.2 cosine distance threshold
                sql = text('''
                    SELECT clean_merchant_name, 
                           embedding::text as target_vector_str, 
                           (embedding <=> CAST(:query_vector AS vector)) AS distance 
                    FROM merchant_knowledge 
                    ORDER BY distance ASC 
                    LIMIT 1
                ''')
                
                try:
                    result = session.execute(sql, {"query_vector": str(query_vector)}).fetchone()
                except Exception as e:
                    logger.warning(f"Vector search failed (pgvector might not be installed): {e}")
                    unmatched_pools[account_id].append(tx)
                    continue
                
                if result and result.distance is not None and result.distance < 0.2:
                    matched_name = result.clean_merchant_name
                    if matched_name not in matched_groups[account_id]:
                        matched_groups[account_id][matched_name] = []
                    matched_groups[account_id][matched_name].append(tx)
                    
                    # VECTOR OBSERVABILITY LOGGER
                    try:
                        target_vec = json.loads(result.target_vector_str)
                        sim_score = 1.0 - float(result.distance)
                        
                        # Truncate 768-D vectors to 5 dimensions for safe JSONL logging
                        trunc_A = [round(x, 4) for x in query_vector[:5]] + ["..."]
                        trunc_B = [round(x, 4) for x in target_vec[:5]] + ["..."]
                        
                        log_payload = {
                            "timestamp": datetime.utcnow().isoformat(),
                            "query_text_A": desc,
                            "target_text_B": matched_name,
                            "cosine_similarity": round(sim_score, 4),
                            "distance": round(float(result.distance), 4),
                            "vector_A_truncated": trunc_A,
                            "vector_B_truncated": trunc_B
                        }
                        
                        with open(log_file_path, "a") as f:
                            f.write(json.dumps(log_payload) + "\n")
                    except Exception as e:
                        logger.error(f"Failed to write vector telemetry: {e}")
                        
                else:
                    unmatched_pools[account_id].append(tx)
                    
        # Transform matched groups into valid clusters (mirroring Phase 3 output format)
        rag_clusters = []
        for acc_id, groups in matched_groups.items():
            for m_name, cluster_txs in groups.items():
                if len(cluster_txs) >= 2:
                    dates = [t.date for t in cluster_txs]
                    dates.sort()
                    amounts = [abs(t.amount) for t in cluster_txs]
                    time_gaps = [(dates[j] - dates[j-1]).days for j in range(1, len(dates))]
                    median_gap = np.median(time_gaps)
                    
                    if median_gap >= 5:
                        std_gap = np.std(time_gaps)
                        if std_gap <= self.time_variance_threshold_days:
                            rag_clusters.append({
                                "anchor_tx": cluster_txs[0],
                                "account_id": acc_id,
                                "transactions": cluster_txs,
                                "time_gaps": time_gaps,
                                "amounts": amounts,
                                "dates": dates,
                                "median_gap": median_gap
                            })
                else:
                    # Not enough txs for a cluster, throw back to unmatched
                    unmatched_pools[acc_id].extend(cluster_txs)
                    
        logger.info(f"RAG Intercept yielded {len(rag_clusters)} immediate semantic clusters.")
        return rag_clusters, unmatched_pools

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
                response = self.chat_model.invoke(prompt, config={"callbacks": [CallbackHandler()], "metadata": {"langfuse_tags": ["subscription-detector"]}})
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

        # Phase 5: Continuous Vector Archiving
        from models.database_models import MerchantKnowledge
        from sqlalchemy.dialects.postgresql import insert
        
        new_knowledge_count = 0
        for c in clusters:
            anchor_tx = c["anchor_tx"]
            
            # Find the subscription we just created/updated to get the merchant name
            sub_hash = hashlib.sha256(f"{c['account_id']}_{self._clean_structural_noise(anchor_tx.semi_cleaned_description or anchor_tx.description).strip().lower()}".encode('utf-8')).hexdigest()
            sub = next((s for s in detected_subscriptions if s.cluster_signature_hash == sub_hash), None)
            
            if sub and sub.merchant_name and sub.merchant_name != "Unknown":
                # Embed the clean merchant name
                try:
                    merchant_vector = self.embeddings.embed_query(sub.merchant_name)
                    
                    # Upsert into MerchantKnowledge
                    stmt = insert(MerchantKnowledge).values(
                        knowledge_uuid=str(uuid.uuid4()),
                        clean_merchant_name=sub.merchant_name,
                        category="Subscription",  # Baseline fallback
                        embedding=merchant_vector,
                        is_human_verified=False,
                        created_at=datetime.utcnow()
                    )
                    
                    # If it already exists by name, we just do nothing (ON CONFLICT DO NOTHING)
                    # Assuming clean_merchant_name is unique, or we just insert it.
                    # Wait, if clean_merchant_name is not unique, we might insert duplicates.
                    # Let's just check if it exists first to be safe and ORM-agnostic
                    exists = session.query(MerchantKnowledge).filter_by(clean_merchant_name=sub.merchant_name).first()
                    if not exists:
                        session.execute(stmt)
                        new_knowledge_count += 1
                except Exception as e:
                    logger.warning(f"Failed to archive MerchantKnowledge for {sub.merchant_name}: {e}")
                    
        if new_knowledge_count > 0:
            session.commit()
            logger.info(f"Continuously Archived {new_knowledge_count} new merchant vectors into pgvector.")

