import os
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class StringCleanerState(TypedDict):
    messages: Sequence[BaseMessage]
    raw_string: str
    cleaned_string: str

class StringCleaningAgent:
    """
    Refactored to be a standalone LangGraph Agent with ChatOpenAI capabilities.
    """
    def __init__(self):
        self.app = self._build_graph()
        
    def _build_graph(self):
        base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")
        if not base_url.endswith("/v1"): 
            base_url = f"{base_url}/v1"
            
        llm = ChatOpenAI(
            base_url=base_url,
            api_key="budai-local",
            model="Qwen3.5-9B-GGUF",
            temperature=0
        )
        
        def process_string(state: StringCleanerState):
            logger.info(f"String Cleaning Agent evaluating: {state['raw_string']}")
            response = llm.invoke(state["messages"])
            return {"messages": [response], "cleaned_string": response.content.strip()}
            
        workflow = StateGraph(StringCleanerState)
        workflow.add_node("process", process_string)
        workflow.set_entry_point("process")
        workflow.add_edge("process", END)
        
        return workflow.compile()
        
    @staticmethod
    def clean_merchant_name(raw_string: str) -> str:
        if not raw_string:
            return "Unknown"
            
        agent = StringCleaningAgent()
        messages = [
            SystemMessage(content='\\n\\n### STRICT ROLE BOUNDARIES & EXPERTISE ###\\nYou are a highly specialized autonomous agent within the BudAI multi-agent network. Your role is strictly isolated to your specific domain of expertise. Do not attempt to execute actions outside your purview. You possess hyper-focused tools designed exclusively for your analytical tasks. Communicate professionally, mathematically, and directly. Avoid conversational filler.' + "You are the String Cleaning Agent. Your sole job is to receive a raw, messy banking transaction description and output ONLY the clean, standard merchant or brand name. Remove all dates, store numbers (e.g. #1234), locations (e.g. LONDON), and bank prefixes (e.g. VIS, POS, CRV). Do not add any conversational text. Output strictly the clean name."),
            HumanMessage(content=f"Clean this transaction string: {raw_string}")
        ]
        
        try:
            final_state = agent.app.invoke({"messages": messages, "raw_string": raw_string, "cleaned_string": ""})
            return final_state["cleaned_string"]
        except Exception as e:
            logger.error(f"StringCleaningAgent LLM failed: {e}")
            return raw_string
