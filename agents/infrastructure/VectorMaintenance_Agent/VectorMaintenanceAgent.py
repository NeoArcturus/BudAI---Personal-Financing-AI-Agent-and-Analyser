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
    """Fetches a list of highly similar merchant vectors that might be duplicates."""
    with SessionLocal() as session:
        query = text("""
            SELECT 
                a.knowledge_uuid as id1, a.clean_merchant_name as name1,
                b.knowledge_uuid as id2, b.clean_merchant_name as name2,
                (a.embedding <=> b.embedding) as distance
            FROM merchant_knowledge a
            JOIN merchant_knowledge b 
              ON a.knowledge_uuid != b.knowledge_uuid 
             AND a.category = b.category
            WHERE (a.embedding <=> b.embedding) < 0.02
            LIMIT 5
        """)
        results = session.execute(query).fetchall()
        
        if not results:
            return "No duplicates found."
            
        dupes = []
        for r in results:
            dupes.append({"id1": r.id1, "name1": r.name1, "id2": r.id2, "name2": r.name2, "distance": float(r.distance)})
        return json.dumps(dupes)

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
            return f"Successfully merged {delete_id} into {keep_id}."
        except Exception as e:
            session.rollback()
            return f"Failed to merge: {str(e)}"

class VectorMaintenanceAgent:
    """
    Refactored to be a standalone LangGraph Agent with ChatOpenAI capabilities.
    """
    def __init__(self):
        self.app = self._build_graph()
        
    def _build_graph(self):
        tools = [fetch_duplicate_candidates, merge_duplicate_vectors]
        tool_node = ToolNode(tools)
        
        base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")
        if not base_url.endswith("/v1"): 
            base_url = f"{base_url}/v1"
            
        llm = ChatOpenAI(base_url=base_url, api_key="budai-local", model="Qwen3.5-9B-GGUF", temperature=0)
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
