import os
import json
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from sqlalchemy import text
from config import SessionLocal
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class VectorState(TypedDict):
    messages: Sequence[BaseMessage]

@tool
def fetch_duplicate_candidates() -> str:
    """Fetches a list of highly similar merchant vectors from Qdrant that might be duplicates."""
    try:
        from services.qdrant_service import get_qdrant_service
        qdrant = get_qdrant_service()
        
        # Scroll points from Qdrant
        points, _ = qdrant.client.scroll(
            collection_name="merchant_knowledge",
            limit=50,
            with_vectors=True,
            with_payload=True
        )
        
        dupes = []
        for p in points:
            if not p.vector:
                continue
            hits = qdrant.search_merchant_knowledge(p.vector, limit=3, score_threshold=0.98)
            for h in hits:
                if str(h["id"]) != str(p.id):
                    dupes.append({
                        "id1": str(p.id),
                        "name1": p.payload.get("clean_merchant_name", ""),
                        "id2": str(h["id"]),
                        "name2": h["payload"].get("clean_merchant_name", ""),
                        "score": float(h.get("score", 0.0))
                    })
            if len(dupes) >= 5:
                break
                
        if not dupes:
            return "No duplicates found."
        return json.dumps(dupes[:5])
    except Exception as e:
        logger.error(f"Error fetching duplicates from Qdrant: {e}")
        return "No duplicates found."

@tool
def merge_duplicate_vectors(keep_id: str, delete_id: str) -> str:
    """Merges two duplicate vector nodes. Points transactions to keep_id and deletes delete_id."""
    with SessionLocal() as session:
        try:
            update_txs = text("UPDATE transactions SET merchant_knowledge_uuid = :keep_id WHERE merchant_knowledge_uuid = :delete_id")
            session.execute(update_txs, {"keep_id": keep_id, "delete_id": delete_id})
            
            delete_node = text("DELETE FROM merchant_knowledge WHERE knowledge_uuid = :delete_id")
            session.execute(delete_node, {"delete_id": delete_id})
            
            session.commit()
            
            # Delete from Qdrant
            try:
                from services.qdrant_service import get_qdrant_service
                from qdrant_client import models
                import uuid
                qdrant = get_qdrant_service()
                try:
                    p_id = str(uuid.UUID(str(delete_id)))
                except ValueError:
                    p_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(delete_id)))
                qdrant.client.delete(
                    collection_name="merchant_knowledge",
                    points_selector=models.PointIdsList(points=[p_id])
                )
            except Exception as q_err:
                logger.warning(f"Failed to delete {delete_id} from Qdrant: {q_err}")

            return f"Successfully merged {delete_id} into {keep_id}."
        except Exception as e:
            session.rollback()
            return f"Failed to merge: {str(e)}"

@tool
def cascade_merchant_relabeling(clean_merchant_name: str, new_category: str, new_sub_category: str, tags: list[str]) -> str:
    """
    Cascades a category change across all identical merchants in PostgreSQL and updates Qdrant vectors.
    """
    with SessionLocal() as session:
        try:
            update_mk = text("""
                UPDATE merchant_knowledge
                SET category = :cat, sub_category = :sub_cat, tags = :tags::jsonb, is_human_verified = TRUE
                WHERE clean_merchant_name = :name
            """)
            session.execute(update_mk, {
                "cat": new_category,
                "sub_cat": new_sub_category,
                "tags": json.dumps(tags),
                "name": clean_merchant_name
            })
            
            update_txs = text("""
                UPDATE transactions
                SET sub_category = :sub_cat, tags = :tags::jsonb
                WHERE merchant_knowledge_uuid IN (
                    SELECT knowledge_uuid FROM merchant_knowledge WHERE clean_merchant_name = :name
                ) OR semi_cleaned_description ILIKE :name_like
            """)
            session.execute(update_txs, {
                "sub_cat": new_sub_category,
                "tags": json.dumps(tags),
                "name": clean_merchant_name,
                "name_like": f"%{clean_merchant_name.lower()}%"
            })
            session.commit()

            try:
                from services.qdrant_service import get_qdrant_service
                from services.embedding_service import get_embedding_service
                qdrant = get_qdrant_service()
                embedder = get_embedding_service()
                rich_context = f"{clean_merchant_name} | {new_category} | {new_sub_category} | {', '.join(tags)}"
                new_vec = embedder.embed_query(rich_context)
                
                anchor_row = session.execute(
                    text("SELECT knowledge_uuid FROM merchant_knowledge WHERE clean_merchant_name = :name"),
                    {"name": clean_merchant_name}
                ).fetchone()
                if anchor_row:
                    k_uuid = anchor_row[0]
                    qdrant.upsert_merchant(
                        knowledge_uuid=k_uuid,
                        clean_merchant_name=clean_merchant_name,
                        category=new_category,
                        sub_category=new_sub_category,
                        tags=tags,
                        vector=new_vec,
                        is_human_verified=True
                    )
            except Exception as q_err:
                logger.warning(f"Could not update Qdrant during cascade: {q_err}")

            return f"Successfully cascaded {clean_merchant_name} to {new_category}."
        except Exception as e:
            session.rollback()
            return f"Failed to cascade relabeling: {str(e)}"

tools = [fetch_duplicate_candidates, merge_duplicate_vectors, cascade_merchant_relabeling]

class VectorMaintenanceAgent:
    """
    Refactored to be a standalone LangGraph Agent with ChatOpenAI capabilities.
    """
    def __init__(self):
        self.app = self._build_graph()
        
    def _build_graph(self):
        tool_node = ToolNode(tools)
        
        from services.llm_factory import get_chat_llm
        llm = get_chat_llm(temperature=0)
        llm_with_tools = llm.bind_tools(tools)
        
        def evaluate_node(state: VectorState):
            logger.info("VectorMaintenanceAgent evaluating RAG memory state...")
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}
            
        def should_continue(state: VectorState):
            messages = state["messages"]
            last_message = messages[-1]
            if not last_message.tool_calls:
                return "end"
            return "continue"
            
        workflow = StateGraph(VectorState)
        workflow.add_node("evaluator", evaluate_node)
        workflow.add_node("tools", tool_node)
        
        workflow.set_entry_point("evaluator")
        workflow.add_conditional_edges("evaluator", should_continue, {"continue": "tools", "end": END})
        workflow.add_edge("tools", "evaluator")
        
        return workflow.compile()
        
    @staticmethod
    def optimize_rag_memory():
        logger.info("Starting Agentic RAG Vector Optimization...")
        agent = VectorMaintenanceAgent()
        
        messages = [
            SystemMessage(content='\\n\\n### STRICT ROLE BOUNDARIES & EXPERTISE ###\\nYou are a highly specialized autonomous agent within the BudAI multi-agent network. Your role is strictly isolated to your specific domain of expertise. Do not attempt to execute actions outside your purview. You possess hyper-focused tools designed exclusively for your analytical tasks. Communicate professionally, mathematically, and directly. Avoid conversational filler.' + "You are the Vector Maintenance Agent. Use `fetch_duplicate_candidates` to find duplicate merchant vectors in the RAG database. If duplicates are found, analyze their names. If they represent the same real-world entity, use `merge_duplicate_vectors` to clean the database."),
            HumanMessage(content="Please review the RAG database for duplicates and optimize it.")
        ]
        
        agent.app.invoke({"messages": messages})
        return True
