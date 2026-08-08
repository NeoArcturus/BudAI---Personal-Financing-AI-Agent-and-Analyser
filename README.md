# BudAI: Agentic AI Personal Financing Platform

BudAI is a personal financing platform built with a multi-agent architecture. It performs deterministic financial analysis, retrieval-augmented generation (RAG), and stochastic forecasting. The system uses deep learning for parameter extraction, a C++ core for simulations, and a distributed Python backend for orchestration.

---

## Motivation and Vision

BudAI is driven by a core philosophy: **AI-driven Financial Empowerment**. 
Our goal is to serve students and working professionals by automating the most stressful aspect of their lives—managing, securing, and growing personal finances. By leveraging advanced AI, BudAI removes the friction from financial planning and provides users with a clear, actionable, and intelligent path forward.

---

## 1. Core Architectural Systems

### 1.1 Native Orchestration & Messaging

- **Modular REST Architecture:** Transitioned from monolithic controllers to explicit HTTP-method-based routing (e.g., `controllers/chat/get.py`) connected via FastAPI `APIRouter`.
- **Direct Tool Execution:** The system has transitioned away from legacy distributed runtimes, now executing all MCP tools directly via a unified `MCPBridge`. Tools are routed locally inside the FastAPI worker, wrapping synchronous tool triggers in `asyncio.to_thread()` to ensure execution occurs without blocking the primary event loop.
- **Vercel Data Stream Implementation:** The system implements the Vercel AI SDK protocol for streaming:
    - `0:`: Standard conversational text chunks.
    - `8:`: Out-of-band JSON metadata (e.g., `session_title_update`, `global_refresh_signal`, and `telemetry` for TTFT/Compute tracking).
    - `9:`: Structured tool invocations (e.g., `render_ui_chart`, `ask_user` for HTIL).

### 1.2 Multi-Agent Orchestration (LangGraph)
The analytical logic is governed by a stateful directed acyclic graph (DAG) implemented via **LangGraph**, coordinating specialized sub-agents:
- **Intent Router (Supervisor):** Utilizes `mlx-community/Qwen3.5-4B-4bit` served via **Rapid-MLX** for query classification and delegation.
- **Specialized Worker Agents:** Includes domain-specific agents such as the `Analyser Agent` (data crunching), `Forecaster Agent` (predictive modeling), `Categorizer Agent` (transaction labeling), `Memory Agent` (FAISS interactions), `Market Agent` (macro-economic context), `Scenario Agent` (what-if analysis), and `Health Agent` (financial wellness).
- **Global Inference Lock:** Implements a localized semaphore (`llm_manager.py`) to prevent concurrent FastAPI worker threads from crashing the local LLM with overlapping generation requests.
- **State Schema:** The `BudAIState` manages session persistence, user UUIDs, active account IDs, and a buffer for raw data and chart metadata.
- **Clarification Loop:** Proactively triggers an `ask_user` tool call for Human-in-the-Loop (HTIL) engagement when user intent is underspecified, pausing execution until the user provides clarification.

### 1.3 Quantitative Forecasting Pipeline
Projections are generated via a hybrid computational architecture:
- **Parameter Extraction:** A PyTorch-based **LSTM (Long Short-Term Memory)** network processes a 30-day window of transaction amounts and categories. It extracts Bates model parameters: Mean Reversion Speed ($\kappa$), Long-term Variance ($\theta$), Volatility of Volatility ($\xi$), and Jump Intensity ($\lambda$).
- **Stochastic Core:** The extracted parameters drive a **C++ shared object (.so)** implementing the Bates Model. It runs 1,000 parallel Monte Carlo simulations per account, incorporating recurring bill detection to clamp simulated paths to deterministic historical patterns.

### 1.4 Retrieval-Augmented Generation (RAG)
- **Vector Core:** Uses **FAISS (Facebook AI Similarity Search)** with `IndexFlatL2` for sub-millisecond semantic retrieval, with a planned migration to **pgvector** for strict relational integrity.
- **Embedding Model:** Transactions are vectorized using the **`all-MiniLM-L6-v2`** SentenceTransformer (384-dimensional dense vectors).
- **Contextual Financial Profile (CFP):** Compiles an 8k-token grounding block including:
    - Tier 1: Immediate Liquidity & Net Cash Flow.
    - Tier 2: Recurring Bill Frequency & Active Subscriptions.
    - Tier 3: Merchant Clusters & High-Frequency Categories.
    - Tier 4: RAG-retrieved semantic historical context.

### 1.5 Session Management & Observability
- **Extended Context Memory:** The orchestrator retrieves up to the last 6 messages per session to ensure deep multi-turn context.
- **Session Customization:** Dedicated REST endpoints allow users to seamlessly rename active chat sessions.
- **Execution Telemetry:** Every inference stream automatically calculates its Time-to-First-Token (TTFT), Compute Time, and Token Count. These metrics, alongside the exact "reasoning" chain-of-thought tokens, are persisted to PostgreSQL (`chat_history` table) and streamed to the frontend for UI visibility.

---

## 2. Data Integrity & Management

### 2.1 Ingestion and Deduplication
- **Asynchronous Webhook Ingestion:** Syncing is offloaded to background threads. The system triggers TrueLayer data updates via the `/truelayer` webhook, processing massive payloads asynchronously without blocking the user interface.
- **Database-First Caching & High-Water Marks:** To avoid rate limits, the system tracks the `last_synced_at` timestamp per account. It queries local PostgreSQL first and only hits the TrueLayer API using a 3-day overlap window (`last_synced_at - timedelta(days=3)`) to ensure zero transaction dropping during sync increments.
- **Composite Constraints:** Strict PostgreSQL unique constraints on `(account_id, provider_transaction_id)` mathematically reject redundant TrueLayer API duplicates.
- **PostgreSQL Persistence:** All records are stored in a PostgreSQL 15 cluster with optimized pooling (SQLAlchemy `pool_size=20`, `max_overflow=40`).
- **Lazy ML Categorization:** Incoming transactions are saved instantly during sync, while heavy ML classification is offloaded to a non-blocking background thread.

### 2.2 Advanced Categorization ETL Pipeline
To ensure high accuracy with zero third-party API costs, transactions are categorized via a 4-stage "waterfall" architecture:
1. **SQL Cache:** Instant exact-match lookups against the localized `MerchantKnowledge` table.
2. **Vector RAG:** Semantic similarity matching using `pgvector` to resolve merchant name variations.
3. **Local LLM Zero-Shot:** Internal weight-based inference for identifying broad merchant categories.
4. **Agentic Web Search MCP:** An automated Playwright/Node-based browser search acts as the ultimate fallback for obscure merchants, saving the summarized web context permanently to the database to prevent repeat searches.

### 2.3 Subscription & Pattern Detection
- **Mathematical Interval Grouping:** Groups transactions by `merchant_name` and computes interval statistics ($\Delta t$) to identify recurring payments.
- **Variance Thresholds:** Flags subscriptions deterministically by analyzing the standard deviation of payment gaps ($\sigma_{\text{days}} \le 15$).
- **Normalized Data Structure:** Stripped legacy denormalized columns (`bank_name`, `account_number`) from the `subscriptions` table, enforcing strict relational joins via `bank_uuid` to maintain 3NF (Third Normal Form) integrity.

---

## 3. Technology Stack

### 3.1 Backend & AI Infrastructure
- **Python Framework:** FastAPI 0.110+ (Uvicorn worker model) with modular REST routing and domain-separated controllers.
- **AI Core:** LangChain 0.2+, LangGraph, Ollama, Rapid-MLX.
- **Security:** AES-256 (Fernet) for at-rest encryption of bank tokens.
- **Networking:** Docker-aware routing (Automatic `localhost` -> `budai-db` hostname rewrite inside containers).

### 3.2 Frontend (Client Application)
- **Framework:** Next.js 16, React 19.
- **Data Fetching:** TanStack Query v5 with 5-minute cache TTL.
- **UI Engine:** HeroUI v3 (Compound pattern), Framer Motion for component transitions.
- **Charting:** Chart.js with a modular `CoreChartEngine` that renders specialized payloads from the database cache.

---

## 4. Development and Deployment

### 4.1 Prerequisites
- **Docker Desktop** (mandatory for full cluster deployment).
- **Node.js 20+** (for independent frontend development).

### 4.2 Cluster Initialization
The application is deployed as a 9-container fleet to ensure isolation of high-CPU ML workloads.

```bash
# Environment Configuration (.env must include DATABASE_URL, OLLAMA_BASE_URL, and TrueLayer API Keys)
cp .env.example .env

# Build and Deploy backend services
docker compose build --no-cache
docker compose up -d
```

### 4.3 Local Frontend Execution & Testing
To run the Next.js frontend independently for UI development:
```bash
cd budai-frontend
npm install
npm run dev
```

For running backend unit tests:
```bash
pytest tests/
```

### 4.4 C++ Shared Object Compilation
```bash
cd services/Forecaster_Agent/mathematics/algorithm
g++ -O3 -shared -fPIC -std=c++17 -o ../hybrid_forecaster.so algorithm.cpp hybrid_algorithm.cpp
```

### 4.5 Database Inspection
```bash
docker exec -it budai-db psql -U postgres -d budai
```

---

## 5. Engineering Standards
- **Deterministic Grounding:** The AI is strictly prohibited from extrapolating or inventing metrics. It enforces a "Strict Data Boundary" where any missing metric results in a "Data Unavailable" response.
- **Privacy First:** All PII and financial data remains within the localized Docker infrastructure. No data is transmitted to external LLM providers.
- **Type Integrity:** Full TypeScript and Pydantic coverage ensuring schema-level enforcement of all data structures moving across the microservices bus.

---

## 6. API Documentation

Interactive API documentation is automatically generated by FastAPI. Once the cluster is running, the OpenAPI schema and Swagger UI can be accessed via:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

## 7. Technical Roadmap & Scaling Strategy

As BudAI prepares to scale to 20,000+ active users, the architecture is evolving from a localized monolithic application to a distributed micro-services architecture:

- **4-Stage Categorization ETL:** Full deployment of the `MerchantKnowledge` pgvector cache combined with the Agentic Web Search MCP (Playwright/Node) to mathematically eliminate LLM hallucinations on obscure transactions.
- **Asynchronous Task Queues:** Migrating heavy ML workloads (PyTorch Monte Carlo simulations) out of the FastAPI ASGI event loop into a dedicated **Celery** cluster using Redis as the message broker.
- **Connection Pooling:** Implementing **PgBouncer** and transitioning to asynchronous database drivers (`asyncpg`) to prevent SQLAlchemy connection starvation under high concurrency.
- **Multi-Node LLM Inference:** Scaling the current local inference (`host.docker.internal`) into a distributed **vLLM Kubernetes cluster** with continuous batching and HAProxy load balancing.
- **API Resource Management:** Developing token quota systems and strict rate limiting to throttle local LLM consumption per user and prevent infrastructure overload.
- **Advanced Budgeting & Forecasting:** Implementing variance analytics, deterministic 30-day cash flow arrays, and automated paycheck allocation rules.

---

## 8. Future Feature Horizons

Beyond the foundational architecture, BudAI is laying the groundwork for advanced financial intelligence in upcoming releases:

- **Proactive Cash Flow Modeling:** Anticipating future liquidity based on deterministic schedules and predictive modeling.
- **External Market Context:** Integrating macroeconomic indicators and localized financial parameters for demographic benchmarking.
- **Credit Health Optimization:** Simulating debt pay-down strategies and managing long-term liability costs.
- **System Personalization:** Developing multi-category tagging and customized reporting logic tailored to individual user behaviors.
