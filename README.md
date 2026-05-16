# BudAI: The "Digital Twin" Financial Intelligence System

**BudAI** is a high-performance, full-stack personal finance assistant that has evolved from a reactive dashboard into a proactive, context-aware "Digital Twin." It utilizes a multi-agent AI orchestration layer, deep learning, semantic memory, and quantitative mathematics to provide hallucination-resistant analysis, highly personalized forecasting, and interactive visualizations.

---

## Core Capabilities: The AI Upgrade

- **Neuro-Stochastic Forecasting:** A cutting-edge hybrid projection engine. A **PyTorch LSTM** extracts your "Financial DNA" (volatility, jump frequency) from transaction sequences, which dynamically drives a high-speed **C++ Monte Carlo simulation** (Bates Model) mathematically clamped to your live balance.
- **Semantic Memory (RAG):** Multi-year pattern recognition powered by a local **FAISS Vector Database**. Every transaction is semantically embedded (`all-MiniLM-L6-v2`), allowing the AI to recall your historical travel habits, hidden fee structures, and seasonal spending spikes.
- **Asynchronous Learning Loop:** An active learning categorizer (XGBoost). When you manually correct a transaction label, FastAPI `BackgroundTasks` instantly retrains the model, cleanses your entire historical ledger, recalibrates your LSTM forecasting parameters, and re-indexes your semantic memory—all without freezing the UI.
- **Multi-Resolution Financial Profile (MRFP):** To prevent LLM hallucinations, BudAI builds an 8k-token dense context block (Liquidity, Recurring Rhythm, Merchant Footprint, RAG Narrative) and injects it directly into the LLM system prompt.
- **Persistent Chat Sessions:** Seamless conversation management across sessions, allowing you to pick up complex financial planning discussions exactly where you left off.
- **Advisor HUD & Instant Sync:** Real-time localized 2-3 sentence widget insights formatted strictly in GBP (£) and SHA-256 deduplicated, zero-latency transaction syncing via **TrueLayer**.

---

## Tech Stack

### Backend (The Core)
- **Framework:** FastAPI (Python 3.10+)
- **AI Orchestration:** LangGraph, LangChain, Ollama (Local LLM - `qwen3:4b`, `qwen3.5:4b`)
- **Deep Learning & Math:** PyTorch (LSTM), scikit-learn (HistGradientBoosting), C++ (Shared Objects for Bates MC Engine)
- **Semantic Storage (RAG):** FAISS (Facebook AI Similarity Search), SentenceTransformers
- **Database:** SQLite + SQLAlchemy ORM
- **Task Scheduling:** APScheduler (Token rotation), FastAPI BackgroundTasks (ML Learning Loop)

### Frontend (The Narrator)
- **Framework:** Next.js 16 (App Router), React 19
- **Styling:** Tailwind CSS 4, HeroUI v3 (Beta)
- **Visualization:** Chart.js with `CoreChartEngine`
- **State Management:** TanStack Query (Data Fetching), Liquid State Dashboard Architecture

---

## Project Structure

```text
.
├── main.py                 # FastAPI Entrypoint & Background Schedulers
├── config.py               # Security, DB, & Environment Configuration
├── controllers/            # API Routers (Auth, Accounts, Chat, Media, Categorizer, Advisor)
├── models/                 # SQLAlchemy Database Models (Users, Transactions, ChatSessions, ForecastParameters)
├── schemas/                # Pydantic Validation Schemas
├── services/
│   ├── orchestrator_graph.py # LangGraph Multi-Agent Logic & Intent Routing
│   ├── profile_builder.py  # Compiles the Multi-Resolution Financial Profile (MRFP)
│   ├── memory_service.py   # FAISS Vector DB Management & RAG Search
│   ├── workers/            # Specialized AI Worker Nodes (Analyser, Forecaster, Memory, etc.)
│   ├── api_integrator/     # TrueLayer OAuth & Deduplicated Ingestion Pipeline
│   ├── Categorizer_Agent/  # XGBoost Categorization, Preprocessing & Training
│   └── Forecaster_Agent/   # PyTorch LSTM Models & C++ Monte Carlo Engine Bridge
├── mcp_servers/            # Model Context Protocol (MCP) tool servers (Memory, Macro, etc.)
└── budai-frontend/         # Next.js Application
```

---

## Setup and Installation

### 1. Prerequisites
- Python 3.10+
- Node.js 20+
- C++ Compiler (GCC/Clang) for building the forecasting engine
- [Ollama](https://ollama.com/) installed and running locally

### 2. Backend Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Compile C++ Forecasting Engine
cd services/Forecaster_Agent/mathematics/algorithm
g++ -O3 -shared -fPIC -std=c++17 -o ../hybrid_forecaster.so algorithm.cpp hybrid_algorithm.cpp
cd ../../../..

# Configure Environment
cp .env.example .env # Add your TrueLayer Credentials & ENCRYPTION_KEY

# Pull required LLM models
ollama pull qwen3:4b
ollama pull qwen3.5:4b

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

### 3. Frontend Setup
```bash
cd budai-frontend
npm install
npm run dev
```

---

## Security & Architecture Standards

- **Hardware Acceleration:** All ML tools (SentenceTransformers, LSTM, XGBoost) dynamically detect and utilize GPU/MPS/CUDA for maximum performance.
- **Token Encryption:** Bank-access tokens are encrypted at rest using Fernet (AES-128).
- **Local AI Privacy:** All LLM processing and Vector DB (FAISS) storage occurs locally, ensuring sensitive financial histories never leave your infrastructure.
- **Strict Grounding:** The AI acts strictly on the deterministic data provided by the C++ engine and MRFP block, ensuring zero numerical hallucinations.
