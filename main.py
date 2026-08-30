import json
import logging
import asyncio
import os
import warnings

if os.environ.get("COLLECTOR_ENDPOINT"):
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

        from openinference.instrumentation.langchain import LangChainInstrumentor
        from opentelemetry.instrumentation.logging import LoggingInstrumentor
        
        resource = Resource.create({"service.name": "budai-api"})
        
        # gRPC requires stripping the http:// prefix
        base_endpoint = os.environ.get("COLLECTOR_ENDPOINT", "http://signoz-ingester-1:4317").replace("http://", "")
        
        # Traces
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=base_endpoint, insecure=True)))
        trace.set_tracer_provider(tracer_provider)
        
        # Logs
        logger_provider = LoggerProvider(resource=resource)
        logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter(endpoint=base_endpoint, insecure=True)))
        set_logger_provider(logger_provider)
        
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        LoggingInstrumentor().instrument(set_logging_format=True)
    except Exception as e:
        print(f"Failed to initialize SigNoz OpenTelemetry: {e}")

warnings.filterwarnings("ignore", message=".*extra_body.*")

import langchain_openai.chat_models.base as base

original_convert_delta = base._convert_delta_to_message_chunk

def patched_convert_delta(_dict, default_class):
    chunk = original_convert_delta(_dict, default_class)
    if "reasoning_content" in _dict and _dict["reasoning_content"] is not None:
        chunk.additional_kwargs["reasoning_content"] = _dict["reasoning_content"]
    return chunk

base._convert_delta_to_message_chunk = patched_convert_delta
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from config import engine, DATABASE_URL, ALLOWED_ORIGINS, SessionLocal
from middleware.cache_middleware import StripCacheControlMiddleware
from routes.auth_routes import auth_router, callback_router
from routes.account_routes import router as account_router
from routes.chat_routes import router as chat_router
from routes.media_routes import router as media_router
from routes.categorizer_routes import router as categorizer_router
from routes.advisor_routes import router as advisor_router
from routes.market_routes import router as market_router
from routes.memory_routes import router as memory_router
from routes.analytics_routes import router as analytics_router
from services.api_integrator.access_token_generator import AccessTokenGenerator
from services.db_service import init_db
from services.mcp_bridge import MCPBridge
from services.logger_setup import get_core_logger
from models.database_models import Bank

logger = get_core_logger(__name__)

init_db()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info({"message": f"INITIALIZING BUDAI CORE ENGINE", "status_code": 200})
    bridge = MCPBridge()
    FastAPICache.init(InMemoryBackend(), prefix="budai-cache")
    logger.info({"message": f"Background tasks have been offloaded to Prefect workers.", "status_code": 200})
    yield
    logger.info({"message": f"Shutting down core engine", "status_code": 200})


app = FastAPI(title="BudAI API Core", version="2.0.0", lifespan=lifespan)

if os.environ.get("COLLECTOR_ENDPOINT"):
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        SQLAlchemyInstrumentor().instrument(engine=engine)
    except Exception as e:
        print(f"Failed to instrument FastAPI/SQLAlchemy: {e}")

app.add_middleware(StripCacheControlMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Vercel-AI-Data-Stream"],
)

from routes.webhook_routes import router as webhook_router
from routes.widget_routes import router as widget_router
from routes.intent_routes import router as intent_router
from routes.dashboard_routes import router as dashboard_router
from routes.onboarding_routes import router as onboarding_router

app.include_router(auth_router)
app.include_router(callback_router)
app.include_router(account_router)
app.include_router(chat_router)
app.include_router(media_router)
app.include_router(categorizer_router)
app.include_router(advisor_router)
app.include_router(market_router)
app.include_router(webhook_router)
app.include_router(memory_router)
app.include_router(analytics_router)
app.include_router(widget_router)
app.include_router(intent_router)
app.include_router(dashboard_router)
app.include_router(onboarding_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
