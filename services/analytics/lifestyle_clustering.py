import os
import hdbscan
import numpy as np
import uuid
from typing import List, Dict, Any
from datetime import datetime
from sqlmodel import select
import json
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from config import SessionLocal
from models.database_models import Transaction, UserLifestyleProfile, LifestyleCluster
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class ClusterProfile(BaseModel):
    hdbscan_cluster_id: int
    micro_archetype: str
    behavioral_summary: str

class LifestyleAnalysisOutput(BaseModel):
    macro_persona: str
    clusters: List[ClusterProfile]

class LifestyleClusteringService:
    def __init__(self):
        base_url = os.getenv("LLM_BASE_URL", "http://host.docker.internal:8000/v1")
        if not base_url.endswith("/v1"): 
            base_url = f"{base_url}/v1"
            
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.llm = ChatOpenAI(
            model="mlx-community/Qwen3.5-4B-4bit", 
            base_url=base_url, 
            api_key="budai-local", 
            temperature=0,
            max_tokens=1000
        )
        
    def analyze_user_lifestyle(self, user_uuid: str):
        logger.info(f"Starting HDBSCAN lifestyle analysis for user {user_uuid}")
        
        with SessionLocal() as session:
            txs = session.execute(
                select(Transaction).where(Transaction.user_uuid == user_uuid).where(Transaction.amount < 0)
            ).scalars().all()
            
            if not txs or len(txs) < 10:
                logger.warning(f"Not enough transactions to cluster for {user_uuid}")
                return
                
            # Extract strings to embed (Prefer sub_category, fallback to semi_cleaned, fallback to raw)
            def get_cluster_string(tx):
                if tx.sub_category and tx.sub_category.strip():
                    return tx.sub_category.strip()
                if getattr(tx, 'semi_cleaned_description', None) and tx.semi_cleaned_description.strip():
                    return tx.semi_cleaned_description.strip()
                return tx.description or ""
                
            for tx in txs:
                tx._cluster_string = get_cluster_string(tx)

            unique_descriptions = list(set([tx._cluster_string for tx in txs if tx._cluster_string]))
            if not unique_descriptions:
                return
                
            # 1. Vector Extraction
            logger.info(f"Unique clustering strings count: {len(unique_descriptions)}")
            desc_vectors = self.embeddings.embed_documents(unique_descriptions)
            logger.info(f"Vectors returned: {len(desc_vectors)}")
            desc_to_vec = {desc: vec for desc, vec in zip(unique_descriptions, desc_vectors)}
            
            # Map back to transactions
            tx_vectors = []
            valid_txs = []
            for tx in txs:
                if tx._cluster_string and tx._cluster_string in desc_to_vec:
                    vec = desc_to_vec[tx._cluster_string]
                    if not np.isnan(vec).any():
                        tx_vectors.append(vec)
                        valid_txs.append(tx)
                    
            if not tx_vectors:
                logger.warning(f"All generated vectors were invalid (NaNs) for user {user_uuid}")
                return
                
            X = np.array(tx_vectors)
            logger.info(f"X shape constructed: {X.shape}")
        
        # 2. Cluster Generation
        clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric='euclidean')
        cluster_labels = clusterer.fit_predict(X)
        outlier_scores = clusterer.outlier_scores_
        
        # 3. Data Aggregation & Anomaly Tagging
        clusters_data = {}
        with SessionLocal() as session:
            for idx, tx in enumerate(valid_txs):
                label = int(cluster_labels[idx])
                score = float(outlier_scores[idx]) if outlier_scores is not None else 0.0
                
                if score > 0.95:
                    # Tag semantic anomaly directly in the DB
                    db_tx = session.execute(select(Transaction).where(Transaction.transaction_uuid == tx.transaction_uuid)).scalars().first()
                    if db_tx:
                        db_tx.is_semantic_anomaly = True
                        session.add(db_tx)
                
                if label == -1:
                    continue # Noise
                    
                if label not in clusters_data:
                    clusters_data[label] = {
                        "txs": [],
                        "total_spend": 0.0,
                        "merchants": set()
                    }
                clusters_data[label]["txs"].append(tx)
                clusters_data[label]["total_spend"] += abs(tx.amount) if tx.amount else 0.0
                clusters_data[label]["merchants"].add(getattr(tx, '_cluster_string', tx.description))
                
            session.commit()
            
        # 4. Filter to top 5 clusters by a composite score (Density + Financial Value)
        # Formula: (Transaction Count * 50) + Total Spend
        sorted_clusters = sorted(
            clusters_data.items(), 
            key=lambda item: (len(item[1]["txs"]) * 50) + item[1]["total_spend"], 
            reverse=True
        )[:5]
        
        if not sorted_clusters:
            return
            
        cluster_summaries = []
        from collections import Counter
        for label, data in sorted_clusters:
            # Count merchant frequencies
            desc_counts = Counter(getattr(tx, '_cluster_string', tx.description) for tx in data["txs"] if getattr(tx, '_cluster_string', tx.description))
            # Sort by frequency (descending) and then alphabetically (ascending) for strict determinism
            sorted_merchants = sorted(desc_counts.keys(), key=lambda k: (-desc_counts[k], k))
            merchants = sorted_merchants[:7]
            
            # Extract top categories
            cat_counts = Counter(tx.category for tx in data["txs"] if getattr(tx, 'category', None))
            top_cats = [cat for cat, _ in cat_counts.most_common(2)]
            
            # Temporal Context (Day of Week & Time of Day)
            txs_with_dates = [tx for tx in data["txs"] if getattr(tx, 'date', None)]
            if txs_with_dates:
                weekends = sum(1 for tx in txs_with_dates if tx.date.weekday() >= 5)
                day_type = "Mostly Weekends" if weekends > (len(txs_with_dates) / 2) else "Mostly Weekdays"
                
                hours = [tx.date.hour for tx in txs_with_dates]
                avg_hour = sum(hours) / len(hours) if hours else 12
                if 5 <= avg_hour < 12: time_str = "Mornings"
                elif 12 <= avg_hour < 17: time_str = "Afternoons"
                elif 17 <= avg_hour < 22: time_str = "Evenings"
                else: time_str = "Late Night"
            else:
                day_type = "Mixed Days"
                time_str = "Mixed Times"
                
            summary_line = (
                f"Cluster ID {label}: {len(data['txs'])} txs, £{data['total_spend']:.2f} total spend. "
                f"Merchants: {', '.join(merchants)}. "
                f"Categories: {', '.join(top_cats) if top_cats else 'None'}. "
                f"Timing: {day_type}, {time_str}."
            )
            cluster_summaries.append(summary_line)
            
        # 5. Income Context for Macro-Persona Constraint
        with SessionLocal() as session:
            income_txs = session.execute(
                select(Transaction).where(Transaction.user_uuid == user_uuid).where(Transaction.amount > 0)
            ).scalars().all()
            
            if income_txs:
                income_summaries = [f"- £{tx.amount:.2f} from {tx.description} on {tx.date.strftime('%Y-%m-%d') if tx.date else 'Unknown'}" for tx in income_txs]
            else:
                income_summaries = ["No income transactions recorded."]
            income_context = "\n".join(income_summaries)

        # 6. Macro-Persona & Micro-Lifestyle LLM Extraction
        prompt = f"""
        You are a financial behavioral psychologist. Analyze the following spend clusters and income data for a user.
        
        Income Transactions:
        {income_context}
        
        Spend Clusters:
        {chr(10).join(cluster_summaries)}
        
        1. Classify the user into one primary Macro-Persona EXACTLY from this list: [STUDENT, PROFESSIONAL, BUSINESS, RETIREE, CREATIVE].
        2. For each Cluster ID, generate a 2-word micro-archetype (e.g. 'Food Lover', 'Late-Night Cabs', 'Generous Friend', 'Frequent Flyer') and a short, simple 1-sentence summary.
        
        CRITICAL RULES:
        - If the user's Income Transactions are NOT regular (e.g., highly sporadic dates, wildly varying amounts, or no income at all), the Macro-Persona MUST be 'STUDENT'. They CANNOT be 'PROFESSIONAL', 'BUSINESS', or 'RETIREE'.
        - Use simple, everyday conversational language. 
        - DO NOT use academic, financial, or corporate jargon (e.g., avoid words like 'utilize', 'mobility', 'management', 'wealth', 'networking').
        - Make the archetypes highly diverse, casual, and relatable to normal human habits.
        
        Return the result EXACTLY as JSON matching this schema:
        {{
            "macro_persona": "STRING",
            "clusters": [
                {{"hdbscan_cluster_id": INT, "micro_archetype": "STRING", "behavioral_summary": "STRING"}}
            ]
        }}
        """
        
        structured_llm = self.llm.with_structured_output(LifestyleAnalysisOutput)
        from services.llm_manager import GlobalLLMManager
        try:
            GlobalLLMManager.acquire_background()
            try:
                analysis: LifestyleAnalysisOutput = structured_llm.invoke(prompt)
            finally:
                GlobalLLMManager.release()
        except Exception as e:
            logger.error(f"LLM failed to generate lifestyle analysis: {e}")
            return
            
        # 6. Persistence
        with SessionLocal() as session:
            # Upsert UserLifestyleProfile
            profile = session.execute(select(UserLifestyleProfile).where(UserLifestyleProfile.user_uuid == user_uuid)).scalars().first()
            if not profile:
                profile = UserLifestyleProfile(profile_uuid=str(uuid.uuid4()), user_uuid=user_uuid)
            
            profile.macro_persona = analysis.macro_persona
            profile.last_updated = datetime.utcnow()
            session.add(profile)
            
            # Clear old clusters and insert new ones
            old_clusters = session.execute(select(LifestyleCluster).where(LifestyleCluster.user_uuid == user_uuid)).scalars().all()
            for oc in old_clusters:
                session.delete(oc)
                
            for c_info in analysis.clusters:
                cluster_label = c_info.hdbscan_cluster_id
                if cluster_label in clusters_data:
                    data = clusters_data[cluster_label]
                    lc = LifestyleCluster(
                        cluster_uuid=str(uuid.uuid4()),
                        user_uuid=user_uuid,
                        hdbscan_cluster_id=cluster_label,
                        micro_archetype=c_info.micro_archetype,
                        behavioral_summary=c_info.behavioral_summary,
                        total_spend=data["total_spend"],
                        transaction_count=len(data["txs"]),
                        last_updated=datetime.utcnow()
                    )
                    session.add(lc)
            
            session.commit()
            
        logger.info(f"Successfully completed lifestyle clustering for user {user_uuid}")
