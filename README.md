# BudAI: Agentic AI Personal Financing Platform

BudAI is a personal financing platform built with an agentic AI architecture. It performs deterministic financial analysis, retrieval-augmented generation (RAG), and stochastic forecasting. The system uses deep learning for parameter extraction, a C++ core for simulations, and a distributed Python backend for orchestration.

---

## Motivation and Vision

BudAI is driven by a core philosophy: **AI-driven Financial Empowerment**. 
Our goal is to serve students and working professionals by automating the most stressful aspect of their lives—managing, securing, and growing personal finances. By leveraging advanced AI, BudAI removes the friction from financial planning and provides users with a clear, actionable, and intelligent path forward.

---

## 1. Core Architectural Systems

### 1.1 Native Orchestration & Messaging

- **Direct Tool Execution:** The system has transitioned away from legacy distributed runtimes, now executing all MCP tools directly via a unified `MCPBridge`. Tools are routed locally inside the FastAPI worker, wrapping synchronous tool triggers in `asyncio.to_thread()` to ensure execution occurs without blocking the primary event loop.
- **Vercel Data Stream Implementation:** The system implements the Vercel AI SDK protocol for streaming:
    - `0:`: Standard conversational text chunks.
    - `8:`: Out-of-band JSON metadata (e.g., `session_title_update`, `global_refresh_signal`, and `telemetry` for TTFT/Compute tracking).
    - `9:`: Structured tool invocations (e.g., `render_ui_chart`, `ask_user` for HTIL).

### 1.2 Multi-Agent Orchestration (LangGraph)
The analytical logic is governed by a stateful directed acyclic graph (DAG) implemented via **LangGraph**, coordinating specialized sub-agents:
- **Intent Router (Supervisor):** Utilizes `mlx-community/Qwen3.5-4B-4bit` served via **Rapid-MLX** for query classification and delegation.
- **Specialized Worker Agents:** Includes domain-specific agents such as the `Analyser Agent` (data crunching), `Forecaster Agent` (predictive modeling), `Categorizer Agent` (transaction labeling), `Memory Agent` (FAISS interactions), `Market Agent` (macro-economic context), `Scenario Agent` (what-if analysis), and `Health Agent` (financial wellness).
- **State Schema:** The `BudAIState` manages session persistence, user UUIDs, active account IDs, and a buffer for raw data and chart metadata.
- **Clarification Loop:** Proactively triggers an `ask_user` tool call for Human-in-the-Loop (HTIL) engagement when user intent is underspecified, pausing execution until the user provides clarification.

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

### 1.5 Session Management & Observability
- **Extended Context Memory:** The orchestrator retrieves up to the last 6 messages per session to ensure deep multi-turn context.
- **Session Customization:** Dedicated REST endpoints allow users to seamlessly rename active chat sessions.
- **Execution Telemetry:** Every inference stream automatically calculates its Time-to-First-Token (TTFT), Compute Time, and Token Count. These metrics, alongside the exact "reasoning" chain-of-thought tokens, are persisted to PostgreSQL (`chat_history` table) and streamed to the frontend for UI visibility.

---

## 2. Data Integrity & Management

### 2.1 Ingestion and Deduplication
- **SHA-256 Hashing:** Every transaction is hashed using a composite key (UserUUID + AccountID + Date + Amount + Description) to ensure deduplication across redundant TrueLayer API syncs.
- **PostgreSQL Persistence:** All records are stored in a PostgreSQL 15 cluster with optimized pooling (SQLAlchemy `pool_size=20`, `max_overflow=40`).

### 2.2 Active Learning Feedback Loop
- **Classification:** Transactions are initially labeled via a **HistGradientBoosting (XGBoost)** classifier trained on standardized financial taxonomies.
- **Feedback Retraining:** Manual category overrides trigger an asynchronous background task. The system recalibrates the local model and performs a historical sweep to reconcile existing ledger labels with the new user-verified logic.

---

## 3. Technology Stack

### 3.1 Backend & AI Infrastructure
- **Python Framework:** FastAPI 0.110+ (Uvicorn worker model).
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
- **Privacy First:** All PII and financial data remains within the localized Docker infrastructure. No data is transmitted to external LLM providers.
- **Type Integrity:** Full TypeScript and Pydantic coverage ensuring schema-level enforcement of all data structures moving across the microservices bus.
