import os
import json
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from config import SessionLocal
from sqlalchemy import text
from services.logger_setup import get_core_logger
from langchain_openai import OpenAIEmbeddings

logger = get_core_logger(__name__)

class CategorizerState(TypedDict):
    messages: Sequence[BaseMessage]
    user_uuid: str
    transaction_uuid: str
    merchant_name: str

def get_embeddings_model():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")
    if not base_url.endswith("/v1"): 
        base_url = f"{base_url}/v1"
        
    return OpenAIEmbeddings(
        base_url=base_url,
        model="text-embedding-nomic-embed-text-v1.5",
        api_key="budai-local",
        check_embedding_ctx_length=False
    )

@tool
def search_merchant_knowledge(merchant_name: str) -> str:
    """
    Search the vector database for historically verified categories for similar merchants.
    Provides the best matching merchant and its category.
    """
    embeddings = get_embeddings_model()
    try:
        vector = embeddings.embed_query(merchant_name)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return "Error generating embeddings."
        
    with SessionLocal() as session:
        # pgvector query (using <=> for cosine distance)
        query = text("""
            SELECT knowledge_uuid, clean_merchant_name, category, (embedding <=> :vector) as distance
            FROM merchant_knowledge
            ORDER BY embedding <=> :vector
            LIMIT 3
        """)
        results = session.execute(query, {"vector": str(vector)}).fetchall()
        
        if not results:
            return "No similar merchants found."
            
        output = "Similar merchants found:\n"
        for row in results:
            output += f"- Merchant: {row.clean_merchant_name}, Category: {row.category}, Distance: {row.distance:.4f}, UUID: {row.knowledge_uuid}\n"
            
        return output

@tool
def save_category(transaction_uuid: str, category: str, merchant_knowledge_uuid: str = None) -> str:
    """
    Save the chosen category and merchant_knowledge_uuid to the transaction in the database.
    """
    with SessionLocal() as session:
        update_query = text("""
            UPDATE transactions 
            SET category = :category, merchant_knowledge_uuid = :knowledge_uuid 
            WHERE transaction_uuid = :transaction_uuid
        """)
        session.execute(update_query, {
            "category": category, 
            "knowledge_uuid": merchant_knowledge_uuid,
            "transaction_uuid": transaction_uuid
        })
        session.commit()
        return f"Successfully saved category '{category}' for transaction {transaction_uuid}."

class CategorizerAgent:
    """
    Standalone LangGraph Agent with ChatOpenAI capabilities for categorization.
    """
    def __init__(self):
        self.app = self._build_graph()
        
    def _build_graph(self):
        tools = [search_merchant_knowledge, save_category]
        tool_node = ToolNode(tools)
        
        base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")
        if not base_url.endswith("/v1"): 
            base_url = f"{base_url}/v1"
            
        llm = ChatOpenAI(
            base_url=base_url,
            api_key="budai-local",
            model="Qwen3.5-9B-GGUF",
            temperature=0
        )
        llm_with_tools = llm.bind_tools(tools)
        
        def evaluate_node(state: CategorizerState):
            logger.info(json.dumps({"message": "Categorizer Agent evaluating state.", "status_code": 200}))
            messages = list(state.get("messages", []))
            
            if not any(isinstance(m, SystemMessage) for m in messages):
                detailed_prompt = (
                    "### ROLE: Categorization Specialist\n"
                    "You are the Categorizer Agent for BudAI. Your sole responsibility is to accurately classify financial transactions into the correct spending categories.\n\n"
                    "\n\n### STRICT ROLE BOUNDARIES & EXPERTISE ###\nYou are a highly specialized autonomous agent within the BudAI multi-agent network. Your role is strictly isolated to your specific domain of expertise. Do not attempt to execute actions outside your purview. You possess hyper-focused tools designed exclusively for your analytical tasks. Communicate professionally, mathematically, and directly. Avoid conversational filler.\n\n### STRICT ANTI-HALLUCINATION PROTOCOL ###\n"
                    "1. DO NOT guess categories blindly. You MUST use the `search_merchant_knowledge` tool to query the RAG vector database for historically verified categories for similar merchants.\n"
                    "2. Once you have retrieved the historical context and determined the correct category, you MUST use the `save_category` tool to save the category and the associated merchant_knowledge_uuid to the transaction.\n"
                    "3. Do not invent new categories outside of the standard budget groups.\n"
                    "4. If you absolutely cannot determine a category, default to 'Uncategorised'.\n"
                )
                messages.insert(0, SystemMessage(content=detailed_prompt))
                
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}
            
        def should_continue(state: CategorizerState):
            messages = state["messages"]
            last_message = messages[-1]
            
            if not last_message.tool_calls:
                return "end"
                
            return "continue"
            
        workflow = StateGraph(CategorizerState)
        workflow.add_node("evaluator", evaluate_node)
        workflow.add_node("tools", tool_node)
        
        workflow.set_entry_point("evaluator")
        workflow.add_conditional_edges(
            "evaluator",
            should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        workflow.add_edge("tools", "evaluator")
        
        return workflow.compile()

    # --- LEGACY METHODS FOR API COMPATIBILITY ---
    
    @property
    def valid_categories(self):
        # Stub valid categories so legacy save_manual_label doesn't crash if it checks them
        return ["Groceries", "Transport", "Entertainment", "Bills", "Dining", "Shopping", "Uncategorised"]

    def save_manual_label(self, user_uuid, transaction_uuid, corrected_label):
        with SessionLocal() as session:
            from models.database_models import Transaction
            tx = session.query(Transaction).filter_by(transaction_uuid=transaction_uuid).first()
            if not tx: return
            
            tx.category = corrected_label
            session.commit()

    def train_global(self):
        # Deprecated: RAG Fast-Learning handles this now. Returning success for Prefect cron.
        return {"trained": True, "reason": "Global training deprecated; RAG Fast-Learning enabled.", "samples": 0}
