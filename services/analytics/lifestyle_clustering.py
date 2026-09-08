from langfuse.langchain import CallbackHandler
import os
import re
import hdbscan
import numpy as np
import uuid
from typing import List, Dict, Any
from datetime import datetime
from sqlmodel import select
import json
import hashlib
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from config import SessionLocal
from sqlalchemy import text
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
        base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")
        if not base_url.endswith("/v1"): 
            base_url = f"{base_url}/v1"
            
        self.embeddings = OpenAIEmbeddings(
            base_url=base_url,
            model="text-embedding-nomic-embed-text-v1.5",
            api_key="budai-local",
            check_embedding_ctx_length=False
        )
        
        self.llm = ChatOpenAI(
            model="Qwen3.5-9B-GGUF", # Mac: model="mlx-community/Qwen3.5-4B-4bit", 
            base_url=base_url, 
            api_key="budai-local", 
            temperature=0,
            max_tokens=20000
        )
        
    def _clean_structural_noise(self, text: str) -> str:
        if not text: return ""
        text = text.lower()
        text = re.sub(r'card transaction of .*? issued by', '', text)
        text = re.sub(r'card transaction of .*? to', '', text)
        text = re.sub(r'\(fee:.*?\)', '', text)
        text = re.sub(r'gbp', '', text)
        text = re.sub(r'usd', '', text)
        text = re.sub(r'eur', '', text)
        text = re.sub(r'[0-9]+', '', text)
        return text.strip()
        
    def analyze_user_lifestyle(self, user_uuid: str):
        logger.info(json.dumps({"message": f"Starting HDBSCAN lifestyle analysis for user {user_uuid}", "status_code": 200}))
        
        with SessionLocal() as session:
            # OPTIMIZATION: Pull unique clustered strings and pre-aggregate their totals natively in SQL
            query = text("""
                SELECT 
                    COALESCE(NULLIF(TRIM(semi_cleaned_description), ''), NULLIF(TRIM(sub_category), ''), TRIM(description)) as cluster_string,
                    SUM(ABS(amount)) as total_spend,
                    COUNT(*) as tx_count
                FROM transactions 
                WHERE user_uuid = :user_uuid AND amount < 0
                GROUP BY cluster_string
                HAVING COUNT(*) > 0
            """)
            
            results = session.execute(query, {"user_uuid": user_uuid}).fetchall()
            
            if not results or len(results) < 5:
                logger.warning(json.dumps({"message": f"Not enough unique transactions to cluster for {user_uuid}", "status_code": 400}))
                return
                
            unique_descriptions = []
            spend_map = {}
            count_map = {}
            
            for r in results:
                raw_str = str(r[0]) if r[0] else "Unknown"
                c_str = self._clean_structural_noise(raw_str)
                if not c_str: c_str = "Unknown"
                
                if c_str not in unique_descriptions:
                    unique_descriptions.append(c_str)
                    spend_map[c_str] = float(r[1])
                    count_map[c_str] = int(r[2])
                else:
                    spend_map[c_str] += float(r[1])
                    count_map[c_str] += int(r[2])

            if not unique_descriptions:
                return
                
            # Create a hash of the unique transactions string set
            sorted_desc = sorted(unique_descriptions)
            current_hash = hashlib.sha256("".join(sorted_desc).encode('utf-8')).hexdigest()
            
            # Check if this exact cluster set was already processed
            profile = session.execute(select(UserLifestyleProfile).where(UserLifestyleProfile.user_uuid == user_uuid)).scalars().first()
            if profile and profile.last_cluster_hash == current_hash:
                logger.info(json.dumps({"message": f"Clustering aborted: Identical transaction set for user {user_uuid}", "status_code": 200}))
                return
                
            # Embeddings & HDBSCAN
            try:
                vectors = self.embeddings.embed_documents(unique_descriptions)
                vector_matrix = np.array(vectors)
                
                min_cluster_size = 2
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom')
                cluster_labels = clusterer.fit_predict(vector_matrix)
            except Exception as e:
                logger.error(json.dumps({"message": f"Error running HDBSCAN: {e}", "status_code": 500}))
                return
                
            # Aggregate clusters
            cluster_groups = {}
            for idx, label in enumerate(cluster_labels):
                if label == -1: continue # Ignore noise
                if label not in cluster_groups:
                    cluster_groups[label] = {
                        "strings": [],
                        "total_spend": 0.0,
                        "tx_count": 0
                    }
                c_str = unique_descriptions[idx]
                cluster_groups[label]["strings"].append(c_str)
                cluster_groups[label]["total_spend"] += spend_map.get(c_str, 0.0)
                cluster_groups[label]["tx_count"] += count_map.get(c_str, 0)
                
            if not cluster_groups:
                logger.info(json.dumps({"message": f"No valid clusters formed (all noise) for user {user_uuid}", "status_code": 200}))
                return
                
            # Sort clusters by total spend to prioritize context
            sorted_clusters = sorted(cluster_groups.items(), key=lambda x: x[1]["total_spend"], reverse=True)
            
            # Construct Prompt Payload
            cluster_summaries = []
            for c_id, data in sorted_clusters[:10]: # Top 10 clusters max
                cluster_summaries.append(
                    f"Cluster {c_id}: "
                    f"Spend: £{data['total_spend']:.2f} ({data['tx_count']} transactions). "
                    f"Sample Items: {', '.join(data['strings'][:5])}..."
                )
                
            # OPTIMIZATION: Income Context SQL Aggregation
            inc_query = text("""
                SELECT description, SUM(amount) as inc_total
                FROM transactions
                WHERE user_uuid = :user_uuid AND amount > 0
                GROUP BY description
                ORDER BY inc_total DESC
                LIMIT 5
            """)
            income_results = session.execute(inc_query, {"user_uuid": user_uuid}).fetchall()
            
            if income_results:
                income_summaries = [f"- £{float(r[1]):.2f} from {r[0]}" for r in income_results]
            else:
                income_summaries = ["No income transactions recorded."]
                
            prompt = f"""
            You are an elite behavioral economist. Analyze these semantic transaction clusters and deduce the user's lifestyle macro-persona.
            
            INCOME SOURCES (Top 5):
            {chr(10).join(income_summaries)}
            
            SPENDING CLUSTERS (HDBSCAN Derived):
            {chr(10).join(cluster_summaries)}
            
            Respond EXACTLY with a JSON object matching this schema:
            {{
                "macro_persona": "e.g., Aspiring Yuppie, Frugal Student, Suburban Parent",
                "clusters": [
                    {{
                        "hdbscan_cluster_id": [The integer ID from the input],
                        "micro_archetype": "Short 2-word label for this specific behavior (e.g., Caffeine Addict)",
                        "behavioral_summary": "1 sentence analyzing the psychological driver behind this cluster."
                    }}
                ]
            }}
            """
            
            try:
                response = self.llm.invoke(prompt, config={"callbacks": [CallbackHandler()], "metadata": {"langfuse_tags": ["lifestyle-clustering"]}})
                
                raw_content = response.content
                if "```json" in raw_content:
                    raw_content = raw_content.split("```json")[1].split("```")[0]
                elif "```" in raw_content:
                    raw_content = raw_content.split("```")[1].split("```")[0]
                    
                analysis = LifestyleAnalysisOutput.model_validate_json(raw_content.strip())
                
            except Exception as e:
                logger.error(json.dumps({"message": f"LLM parsing failed for lifestyle clustering: {e}", "status_code": 500}))
                return
                
            # Persist to DB
            if not profile:
                profile = UserLifestyleProfile(
                    profile_uuid=str(uuid.uuid4()),
                    user_uuid=user_uuid
                )
                session.add(profile)
                
            profile.macro_persona = analysis.macro_persona
            profile.last_cluster_hash = current_hash
            profile.last_updated = datetime.utcnow()
            
            # OPTIMIZATION: Bulk Delete Old Clusters
            session.execute(text("DELETE FROM lifestyle_clusters WHERE user_uuid = :uuid"), {"uuid": user_uuid})
                
            for c_info in analysis.clusters:
                cluster_label = c_info.hdbscan_cluster_id
                if cluster_label in cluster_groups:
                    c_data = cluster_groups[cluster_label]
                    lc = LifestyleCluster(
                        cluster_uuid=str(uuid.uuid4()),
                        user_uuid=user_uuid,
                        hdbscan_cluster_id=cluster_label,
                        micro_archetype=c_info.micro_archetype,
                        behavioral_summary=c_info.behavioral_summary,
                        total_spend=c_data["total_spend"],
                        transaction_count=c_data["tx_count"],
                        representative_entities=",".join(c_data["strings"][:5])
                    )
                    session.add(lc)
                    
            session.commit()
            logger.info(json.dumps({"message": f"Successfully updated lifestyle profile for user {user_uuid}", "status_code": 200}))
