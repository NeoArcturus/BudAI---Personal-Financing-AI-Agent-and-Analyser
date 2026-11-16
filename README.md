# BudAI Backend

BudAI is an intelligent personal finance platform that connects to bank accounts, organizes capital into virtual buckets, analyzes spending patterns, and projects future cashflow corridors.

The backend mirrors physical bank accounts into virtual allocations without transferring actual money between bank accounts. This enables automated budgeting for upcoming expenses, protection of savings goals, and cashflow forecasting.

---

## Architecture Overview

### Virtual Money Allocation
BudAI organizes available capital into distinct virtual partitions:
- **Unallocated Funds:** General available cash reflecting real-time liquid balances across connected accounts and currencies.
- **Committed Expenses:** Capital earmarked for upcoming fixed commitments and recurring expenses.
- **Savings Goals:** Priority-ordered targets designed to accumulate capital for future objectives.

When spending occurs, the platform automatically reconciles bank transactions against the appropriate virtual bucket, absorbing routine expenses while shielding higher-priority savings from unexpected deficits.

### Execution Pipelines

The backend operates across two coordinated workflows:

#### Interactive Financial Assistant
- Orchestrated using a stateful LangGraph multi-agent architecture.
- Handles user queries through conversational chat endpoints.
- Coordinates specialist workers to inspect account balances, rebalance virtual buckets, audit liquidity, and retrieve cashflow projections.
- Streams agent reasoning and intermediate tool steps to client interfaces in real time via Server-Sent Events.

#### Event-Driven Background Services
- Managed through background task flows for reliability and automatic retries.
- Runs automatically when Open Banking events or balance syncs occur.
- Reconciles physical balances with virtual allocations, evaluates funding for upcoming expenses, and broadcasts real-time updates over WebSockets.

### Ingestion and Merchant Classification
- **Bank Synchronization:** Securely fetches settled and pending transactions from connected financial institutions through Open Banking APIs.
- **Deduplication:** Verifies transaction uniqueness across accounts to ensure idempotent ingestion.
- **In-Process Semantic Embeddings:** Computes compact text vector representations locally using lightweight embedding models, eliminating third-party API latency.
- **Vector Retrieval:** Matches merchant names against knowledge bases in vector storage to assign standardized categories and merchant profiles.

### Predictive Analytics and Cashflow Forecasting
- **Recurrence Detection:** Identifies recurring subscriptions and variable utility payments using temporal rhythms and merchant classifications.
- **Surge Monitoring:** Tracks rolling daily spend velocity against user baselines to flag unusual spending activity.
- **Habit and Lifestyle Profiling:** Evaluates spending distributions across essential needs versus discretionary categories.
- **Multi-Path Cashflow Simulation:** Combines neural network parameter estimation with a high-performance C++ simulation engine, generating parallel Monte Carlo trajectories to produce 30-day cashflow corridors (expected, conservative, and disciplined bounds).

### Consolidated Data Delivery
A unified dashboard endpoint provides a complete synthesized snapshot in a single request, including account balances, virtual bucket states, spending trends, alerts, and forecast trajectories. This eliminates client-side network roundtrips and ensures atomic state rendering.

---

## Tech Stack

### Web and API Framework
- **FastAPI:** High-performance asynchronous web framework with native request validation and OpenAPI documentation.
- **Uvicorn:** Production ASGI server for asynchronous event handling.
- **Pydantic v2:** Strongly-typed schema definition and runtime data validation.

### Database and Storage
- **PostgreSQL 15+ / Neon PostgreSQL:** Primary relational database with connection pooling and database-level constraints.
- **SQLModel & SQLAlchemy 2.0:** Relational ORM combining Pydantic validation with SQLAlchemy query generation.
- **pgvector:** Vector similarity indexing for embedded financial attributes.

### Agentic Intelligence and Vector Search
- **LangGraph & LangChain:** Stateful graph execution for multi-agent workflows, tool execution, and streaming protocols.
- **FastEmbed:** In-process embedding generation running locally for fast semantic vector creation.
- **Qdrant:** Vector database managing semantic merchant records and similarity matching.
- **Model Context Protocol (MCP):** Structured protocol for decoupling analytical tools from agent cores.

### Machine Learning and High-Performance Compute
- **Custom C++ Simulation Core:** C++17 shared engine running parallel Monte Carlo simulation paths for forward-looking balance projections.
- **FastEmbed & ONNX Runtime:** In-process quantized semantic embeddings with single-threaded bounds for memory efficiency.
- **NumPy & Pandas:** Vectorized tabular data transformations and statistical aggregation.

### Orchestration, Security and Telemetry
- **Open Banking:** Token-based integration with banking data providers.
- **Prefect OSS:** Resilient workflow orchestration for scheduled evaluations and ingestion flows.
- **Redis:** Key-value caching and publish-subscribe messaging for real-time WebSocket distribution.
- **Structured JSON Logging:** Observability logging routed to standard error with contextual metadata.

---

## Directory Structure

```
├── agents/                       # Specialized financial agents and analytical engines
│   └── core_financial/
│       └── Forecaster_Agent/
│           └── mathematics/      # C++ simulation library and neural models
├── controllers/                  # Request handling, validation, and response formatting
├── middleware/                   # Authentication, CORS, and caching middleware
├── models/                       # Database tables, relationships, and status schemas
├── routes/                       # FastAPI endpoint declarations
├── schemas/                      # Pydantic request and response schemas
├── services/                     # Core business services, integrations, and detection pipelines
│   ├── a2a/                      # Agent-to-Agent communication interfaces
│   ├── analytics/                # Spending habits, surge detection, and forecasting logic
│   ├── api_integrator/           # Open Banking data sync and token management
│   ├── mcp_tools/                # Tool implementations exposed to agent graphs
│   └── orchestration/            # Background workflows and automated rebalancing
├── config.py                     # Application configuration and service clients
└── main.py                       # FastAPI application entry point
```

---

## Key API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/healthz` | System health and connectivity check |
| `GET` | `/api/dashboard/snapshot` | Complete application state snapshot in a single request |
| `GET` | `/api/accounts` | Connected accounts, nominal balances, and synchronization status |
| `POST` | `/api/accounts/sync` | Initiates background synchronization with connected institutions |
| `GET` | `/api/accounts/{id}/transactions` | Account-level transactions with optional date range filters |
| `GET` | `/api/accounts/transactions/search` | Cross-account transaction search |
| `GET` | `/api/buckets/` | Virtual bucket balances, targets, and currency breakdowns |
| `POST` | `/api/buckets/` | Creates a new virtual bucket |
| `POST` | `/api/buckets/transfer` | Reallocates funds between virtual buckets |
| `POST` | `/api/chat` | Conversational assistant endpoint with real-time tool event streaming |
| `POST` | `/api/categorizer/labels` | Updates merchant classification and adjusts semantic vector anchors |
| `GET` | `/api/market/fx` | Live foreign exchange reference rates |

---

## Configuration

System configuration is driven by environment variables. Copy the sample file to begin:

```bash
cp .env.example .env
```

Key configuration areas include:
- **Database:** PostgreSQL pooled and direct connection URIs.
- **Security:** Secret keys for JWT signing and token encryption at rest.
- **Open Banking:** Provider credentials, redirect URLs, and API endpoints.
- **Vector Storage and AI:** Vector database endpoints and language model access tokens.
- **Caching and Events:** Redis connection strings and allowed CORS origins.

---

## Local Development

### Prerequisites
- Python 3.11+
- C++ compiler (`g++` or `clang++` with C++17 support)
- PostgreSQL 15+ (or cloud-hosted PostgreSQL)
- Redis (optional, for caching and event broadcasting)

### Setup Environment

```bash
git clone <repository-url>
cd BudAI---Personal-Financing-AI-Agent-and-Analyser

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Compile Simulation Engine

Build the high-performance C++ forecasting library:

```bash
cd agents/core_financial/Forecaster_Agent/mathematics/algorithm
g++ -O3 -shared -fPIC -std=c++17 -o ../hybrid_forecaster.so algorithm.cpp hybrid_algorithm.cpp
cd ../../../../../
```

### Initialize Database

Initialize schema tables and required default partitions:

```bash
python -c "from services.db_service import init_db; init_db()"
```

### Start Server

Launch the development server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

Interactive documentation is available at `http://localhost:8080/docs`.

---

## Verification

Check API health:

```bash
curl http://localhost:8080/healthz
```

Execute the test suite:

```bash
pytest tests/
```
