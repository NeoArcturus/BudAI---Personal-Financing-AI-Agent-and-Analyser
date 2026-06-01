# BudAI: Distributed Microservices Architecture for Financial Intelligence

BudAI is an institutional-grade microservices platform engineered for deterministic financial analysis, retrieval-augmented generation (RAG), and stochastic forecasting. The system integrates deep learning parameter extraction with a high-performance C++ simulation core, orchestrated through a distributed shared-runtime engine.

---

## 1. Core Architectural Systems

### 1.1 Distributed Orchestration & Messaging
The platform utilizes the **iii engine** as a centralized shared runtime, replacing legacy point-to-point REST/SSE networking.
- **WebSocket Data Bus:** Inter-service communication is handled via a persistent WebSocket bus on port `49134`.
- **Hybrid Async-Sync Bridge:** To resolve `asyncio` loop-affinity conflicts in Python 3.12, the `MCPBridge` implements a thread-isolated execution pattern. Synchronous `iii` triggers are wrapped in `asyncio.to_thread()`, ensuring network calls occur in a separate thread context without blocking the primary event loop.
- **Vercel Data Stream Implementation:** The system implements the Vercel AI SDK protocol for high-fidelity streaming:
    - `0:`: Standard conversational text chunks.
    - `8:`: Out-of-band JSON metadata (e.g., `session_title_update`, `global_refresh_signal`).
    - `9:`: Structured tool invocations (e.g., `render_ui_chart`, `render_account_selector`).

### 1.2 Multi-Agent Orchestration (LangGraph)
The analytical logic is governed by a stateful directed acyclic graph (DAG) implemented via **LangGraph**:
- **Intent Router:** Utilizes `Qwen 2.5 7B-Instruct` for high-precision query classification. It performs a heuristic ambiguity check to ensure account-specific queries are routed to a clarification node if context is missing.
- **State Schema:** The `BudAIState` manages session persistence, user UUIDs, active account IDs, and a buffer for raw data and chart metadata.
- **Clarification Loop:** Proactively triggers a `render_account_selector` tool call when user intent is underspecified, enabling multi-select account context without defaulting to ambiguous global states.

### 1.3 Quantitative Forecasting Pipeline
Projections are generated via a hybrid computational architecture:
- **DNA Parameter Extraction:** A PyTorch-based **LSTM (Long Short-Term Memory)** network processes a 30-day window of transaction amounts and categories. It extracts Bates model parameters: Mean Reversion Speed ($\kappa$), Long-term Variance ($\theta$), Volatility of Volatility ($\xi$), and Jump Intensity ($\lambda$).
- **Stochastic Core:** The extracted parameters drive a **C++ shared object (.so)** implementing the Bates Model. It runs 1,000 parallel Monte Carlo simulations per account, incorporating recurring bill detection to clamp simulated paths to deterministic historical patterns.

### 1.4 Retrieval-Augmented Generation (RAG)
- **Vector Core:** Uses **FAISS (Facebook AI Similarity Search)** with `IndexFlatL2` for sub-millisecond semantic retrieval.
- **Embedding Model:** Transactions are vectorized using the **`all-MiniLM-L6-v2`** SentenceTransformer (384-dimensional dense vectors).
- **Contextual Financial Profile (CFP):** Compiles an 8k-token grounding block including:
    - Tier 1: Immediate Liquidity & Net Cash Flow.
    - Tier 2: Recurring Bill Rhythm & Subscription Footprint.
    - Tier 3: Merchant Clusters & High-Velocity Categories.
    - Tier 4: RAG-retrieved semantic historical context.

---

## 2. Data Integrity & Management

### 2.1 Ingestion and Deduplication
- **SHA-256 Hashing:** Every transaction is hashed using a composite key (UserUUID + AccountID + Date + Amount + Description) to ensure 100% deduplication across redundant TrueLayer API syncs.
- **PostgreSQL Persistence:** All records are stored in a PostgreSQL 15 cluster with optimized pooling (SQLAlchemy `pool_size=20`, `max_overflow=40`).

### 2.2 Active Learning Feedback Loop
- **Classification:** Transactions are initially labeled via a **HistGradientBoosting (XGBoost)** classifier trained on standardized financial taxonomies.
- **Feedback Retraining:** Manual category overrides trigger an asynchronous background task. The system recalibrates the local model and performs a historical sweep to reconcile existing ledger labels with the new user-verified logic.

---

## 3. Technology Stack

### 3.1 Backend & AI Infrastructure
- **Python Framework:** FastAPI 0.110+ (Uvicorn worker model).
- **AI Core:** LangChain 0.2+, LangGraph, Ollama, vLLM (MLX).
- **Security:** AES-256 (Fernet) for at-rest encryption of bank tokens.
- **Networking:** Docker-aware routing (Automatic `localhost` -> `budai-db` hostname rewrite inside containers).

### 3.2 Frontend (Client Application)
- **Framework:** Next.js 16, React 19.
- **Data Fetching:** TanStack Query v5 with 5-minute cache TTL.
- **UI Engine:** HeroUI v3 (Compound pattern), Framer Motion for component transitions.
- **Charting:** Chart.js with a modular `CoreChartEngine` that renders specialized payloads from the database cache.

---

## 4. Development and Deployment

### 4.1 Cluster Initialization
The application is deployed as a 9-container fleet to ensure isolation of high-CPU ML workloads.

```bash
# Environment Configuration
cp .env.example .env

# Build and Deploy
docker compose build --no-cache
docker compose up -d
```

### 4.2 C++ Shared Object Compilation
```bash
cd services/Forecaster_Agent/mathematics/algorithm
g++ -O3 -shared -fPIC -std=c++17 -o ../hybrid_forecaster.so algorithm.cpp hybrid_algorithm.cpp
```

### 4.3 Database Inspection
```bash
docker exec -it budai-db psql -U postgres -d budai
```

---

## 5. Engineering Standards
- **Deterministic Grounding:** The AI is strictly prohibited from extrapolating or inventing metrics. It enforces a "Strict Data Boundary" where any missing metric results in a "Data Unavailable" response.
- **Privacy First:** 100% of PII and financial data remains within the localized Docker infrastructure. No data is transmitted to external LLM providers.
- **Type Integrity:** Full TypeScript and Pydantic coverage ensuring schema-level enforcement of all data structures moving across the microservices bus.
