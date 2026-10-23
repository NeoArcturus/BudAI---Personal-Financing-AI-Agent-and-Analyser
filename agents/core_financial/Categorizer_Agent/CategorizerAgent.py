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

from langgraph.graph.message import add_messages

class CategorizerState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_uuid: str
    transaction_uuid: str
    merchant_name: str

def get_embeddings_model():
    from services.embedding_service import get_embedding_service
    return get_embedding_service()

@tool
def bulk_search_merchant_knowledge(merchant_names: list[str]) -> str:
    """
    Search the vector database (Qdrant) for historically verified categories for a list of merchants.
    Provides the best matching merchant and its category for each.
    """
    if not merchant_names:
        return "No merchants provided."

    from services.qdrant_service import get_qdrant_service
    qdrant = get_qdrant_service()
    embeddings = get_embeddings_model()
    unique_names = list(set(merchant_names))
    output = "Search Results:\n"
    
    for name in unique_names:
        try:
            vector = embeddings.embed_query(name)
            hits = qdrant.search_merchant_knowledge(vector, limit=1, score_threshold=0.85)
            if hits:
                hit = hits[0]
                p = hit.get("payload", {})
                mk_uuid = p.get("knowledge_uuid") or hit["id"]
                c_name = p.get("clean_merchant_name") or name
                cat = p.get("category") or "General"
                sub_cat = p.get("sub_category") or ""
                tags = p.get("tags") or []
                score = hit.get("score", 0.0)
                output += f"- '{name}' matches '{c_name}' (UUID: {mk_uuid}, Category: {cat}, Sub-Category: {sub_cat}, Tags: {tags}, Score: {score:.4f})\n"
            else:
                output += f"- '{name}': No close match found.\n"
        except Exception as e:
            output += f"- '{name}': Search failed ({str(e)}).\n"
    return output

@tool
def bulk_save_categories(transactions: list[dict]) -> str:
    """
    Save the chosen categories and merchant_knowledge_uuids for a batch of transactions.
    'transactions' must be a list of dictionaries with keys:
    - 'transaction_uuid' (str)
    - 'category' (str)
    - 'sub_category' (str)
    - 'tags' (list of str)
    - 'merchant_name' (str)
    - 'merchant_knowledge_uuid' (str, optional, empty string if none)
    """
    import uuid
    import json
    from datetime import datetime
    
    success_count = 0
    embeddings = None
    
    with SessionLocal() as session:
        for tx in transactions:
            tx_uuid = tx.get("transaction_uuid")
            category = tx.get("category")
            sub_category = tx.get("sub_category", "")
            tags = tx.get("tags", [])
            merchant_name = tx.get("merchant_name")
            mk_uuid = tx.get("merchant_knowledge_uuid")
            
            if str(mk_uuid).strip().lower() in ["", "none", "null", "undefined"]:
                mk_uuid = None
                
            if not tx_uuid or not category or not merchant_name:
                continue
                
            if mk_uuid:
                # LLM Anti-Hallucination Guardrail: Verify the UUID actually exists
                check_query = text("SELECT 1 FROM merchant_knowledge WHERE knowledge_uuid = :uuid")
                exists = session.execute(check_query, {"uuid": mk_uuid}).fetchone()
                if not exists:
                    logger.warning(json.dumps({"message": f"LLM hallucinated/invalid mk_uuid {mk_uuid} for {merchant_name}. Generating new one.", "status_code": 400}))
                    mk_uuid = None
                else:
                    # Update existing merchant anchor with resolved category
                    update_mk_query = text("""
                        UPDATE merchant_knowledge 
                        SET category = :cat, sub_category = :sub_cat, tags = :tags
                        WHERE knowledge_uuid = :uuid
                    """)
                    session.execute(update_mk_query, {
                        "cat": category,
                        "sub_cat": sub_category,
                        "tags": json.dumps(tags),
                        "uuid": mk_uuid
                    })
                    try:
                        from services.qdrant_service import get_qdrant_service
                        get_qdrant_service().update_merchant_payload(
                            knowledge_uuid=mk_uuid,
                            category=category,
                            sub_category=sub_category,
                            tags=tags
                        )
                    except Exception as e:
                        logger.error(f"Failed to update Qdrant payload for anchor {mk_uuid}: {e}")
                    
            if not mk_uuid:
                try:
                    if not embeddings:
                        embeddings = get_embeddings_model()
                    vec = embeddings.embed_query(merchant_name)
                    mk_uuid = str(uuid.uuid4())
                    
                    insert_query = text("""
                        INSERT INTO merchant_knowledge (knowledge_uuid, clean_merchant_name, category, sub_category, tags, is_human_verified, created_at)
                        VALUES (:uuid, :name, :cat, :sub_cat, :tags, FALSE, :now)
                    """)
                    session.execute(insert_query, {
                        "uuid": mk_uuid,
                        "name": merchant_name,
                        "cat": category,
                        "sub_cat": sub_category,
                        "tags": json.dumps(tags),
                        "now": datetime.utcnow()
                    })

                    from services.qdrant_service import get_qdrant_service
                    get_qdrant_service().upsert_merchant(
                        knowledge_uuid=mk_uuid,
                        clean_merchant_name=merchant_name,
                        category=category,
                        sub_category=sub_category,
                        tags=tags,
                        vector=vec,
                        is_human_verified=False
                    )
                except Exception as e:
                    logger.error(f"Failed to create new merchant knowledge for {merchant_name}: {e}")
                    mk_uuid = None
                    
            update_query = text("""
                UPDATE transactions 
                SET merchant_knowledge_uuid = :knowledge_uuid 
                WHERE transaction_uuid = :transaction_uuid
                   OR semi_cleaned_description = (
                    SELECT semi_cleaned_description 
                    FROM transactions 
                    WHERE transaction_uuid = :transaction_uuid
                )
            """)
            session.execute(update_query, {
                "knowledge_uuid": mk_uuid,
                "transaction_uuid": tx_uuid
            })
            success_count += 1
            
        session.commit()
    return f"Successfully saved {success_count} transactions."

class CategorizerAgent:
    """
    Standalone LangGraph Agent with ChatOpenAI capabilities for categorization.
    """
    def __init__(self):
        self.app = self._build_graph()
        
    def _build_graph(self):
        tools = [bulk_search_merchant_knowledge, bulk_save_categories]
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
                    "1. DO NOT guess categories blindly. You MUST extract all unique merchant names from the batch and pass them as a list to the `bulk_search_merchant_knowledge` tool.\n"
                    "2. Once you have retrieved the historical context, you MUST use the `bulk_save_categories` tool to save ALL transactions in the batch simultaneously.\n"
                    "3. You MUST map EVERY transaction to one of these EXACT categories. Do not invent new ones:\n"
                    "   - Income\n"
                    "   - Housing\n"
                    "   - Food & Dining\n"
                    "   - Transportation\n"
                    "   - Utilities\n"
                    "   - Entertainment & Lifestyle\n"
                    "   - Subscriptions & Digital Services\n"
                    "   - Shopping & Retail\n"
                    "   - Healthcare\n"
                    "   - Transfers & Payments\n"
                    "   - Fees & Charges\n"
                    "   - Savings & Investments\n"
                    "   - Taxes & Government Payments\n"
                    "   - Uncategorized\n"
                    "4. If you absolutely cannot determine a category, default to 'Uncategorized'.\n"
                    "5. You MUST also generate an appropriate `sub_category` (string) and `tags` (list of strings) for EVERY transaction based on its nature.\n"
                    "6. If you do not have a merchant_knowledge_uuid for a transaction, pass exactly \"\" (empty string) for its merchant_knowledge_uuid. Do not pass 'null' or 'None'.\n"
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
        return [
            "Income", "Housing", "Food & Dining", "Transportation", "Utilities", 
            "Entertainment & Lifestyle", "Subscriptions & Digital Services", 
            "Shopping & Retail", "Healthcare", "Transfers & Payments", 
            "Fees & Charges", "Savings & Investments", "Taxes & Government Payments", 
            "Uncategorized"
        ]

    def save_manual_label(self, user_uuid, transaction_uuid, corrected_label):
        with SessionLocal() as session:
            from models.database_models import Transaction, MerchantKnowledge
            import uuid
            from datetime import datetime
            
            tx = session.query(Transaction).filter_by(transaction_uuid=transaction_uuid).first()
            if not tx: return

            # If transaction is linked to a Merchant Anchor, update the anchor directly
            if tx.merchant_knowledge_uuid:
                update_query = text("""
                    UPDATE merchant_knowledge
                    SET category = :cat, is_human_verified = TRUE
                    WHERE knowledge_uuid = :uuid
                """)
                session.execute(update_query, {"cat": corrected_label, "uuid": tx.merchant_knowledge_uuid})
                try:
                    from services.qdrant_service import get_qdrant_service
                    get_qdrant_service().update_merchant_payload(
                        knowledge_uuid=tx.merchant_knowledge_uuid,
                        category=corrected_label,
                        is_human_verified=True
                    )
                except Exception as e:
                    logger.error(f"Failed to update Qdrant payload on manual label: {e}")
                session.commit()
                return
            
            # Find the current merchant_name from the description
            merchant_name = tx.semi_cleaned_description
            if not merchant_name:
                return

            # Check if there is an existing merchant_knowledge row for this name AND this category
            query = text("""
                SELECT knowledge_uuid 
                FROM merchant_knowledge 
                WHERE clean_merchant_name = :name AND category = :cat
                LIMIT 1
            """)
            res = session.execute(query, {"name": merchant_name, "cat": corrected_label}).first()
            
            if res:
                # Target exists, just point to it
                tx.merchant_knowledge_uuid = res.knowledge_uuid
            else:
                # We need to create a new row for this specific merchant name and category combo
                embeddings = get_embeddings_model()
                vec = embeddings.embed_query(merchant_name)
                
                new_mk_uuid = str(uuid.uuid4())
                
                insert_query = text("""
                    INSERT INTO merchant_knowledge (knowledge_uuid, clean_merchant_name, category, sub_category, tags, is_human_verified, created_at)
                    VALUES (:uuid, :name, :cat, NULL, '[]'::json, TRUE, :now)
                """)
                session.execute(insert_query, {
                    "uuid": new_mk_uuid,
                    "name": merchant_name,
                    "cat": corrected_label,
                    "now": datetime.utcnow()
                })

                from services.qdrant_service import get_qdrant_service
                get_qdrant_service().upsert_merchant(
                    knowledge_uuid=new_mk_uuid,
                    clean_merchant_name=merchant_name,
                    category=corrected_label,
                    vector=vec,
                    tags=[],
                    is_human_verified=True
                )
                
                tx.merchant_knowledge_uuid = new_mk_uuid
                
            session.commit()

    def train_global(self):
        # Deprecated: RAG Fast-Learning handles this now. Returning success for Prefect cron.
        return {"trained": True, "reason": "Global training deprecated; RAG Fast-Learning enabled.", "samples": 0}
