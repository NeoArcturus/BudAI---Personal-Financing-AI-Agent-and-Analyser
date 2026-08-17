# BudAI: Personal Finance Assistant

BudAI is a personal finance platform that uses AI and quantitative models to help users manage their money. It connects to bank accounts, categorizes transactions, runs financial simulations, and provides an interactive dashboard tailored to the user's specific financial situation.

## How It Works

### 1. User Profiles and Onboarding
When users register, they provide basic demographic data and financial goals. A local AI model analyzes this input and assigns a **Persona** (e.g., STUDENT, PROFESSIONAL, BUSINESS, RETIREE, CREATIVE). 
To keep the database lightweight, the backend does not store UI layout data. Instead, it dynamically generates a customized dashboard layout for the frontend based on the user's Persona every time they log in.

### 2. Transaction Syncing and Categorization
BudAI connects to banks via TrueLayer. Transactions are synced in the background using a 3-day overlap to prevent missing data. Once downloaded, transactions go through a 4-step categorization process without relying on expensive external APIs:
1. **Direct Match:** Checks local database for known merchants.
2. **Vector Search:** Uses semantic search (`pgvector`/FAISS) to find similar merchants.
3. **Local AI:** Uses a local LLM to guess the category for unknown merchants.
4. **Web Search:** Falls back to an automated browser search to figure out obscure charges.

### 3. Financial Forecasting
BudAI goes beyond simple budgeting by running actual financial simulations:
- A PyTorch LSTM network looks at the last 30 days of spending to extract mathematical parameters.
- A custom C++ core runs 1,000 parallel Monte Carlo simulations per account to predict future balances and flag potential overdrafts before they happen.

### 4. Interactive AI Chat
The chat interface has programmatic access to the dashboard. The AI can highlight charts, filter transaction data globally, and stream interactive widgets (like a debt payoff slider) directly into the chat window using the Vercel AI SDK.

---

## Technology Stack

- **Backend:** FastAPI (Python), SQLAlchemy, PostgreSQL
- **Frontend:** Next.js (React 19), HeroUI, Chart.js, TanStack Query
- **AI & Data:** LangGraph, Ollama (Local Inference), PyTorch, FAISS/pgvector
- **Simulations:** Custom C++ shared objects (`.so`)

## Project Architecture

The codebase follows strict RESTful standards and modular design:
- **Controllers:** Separated by resource and HTTP method (e.g., `controllers/auth/get.py`, `controllers/chat/post.py`) for maintainability.
- **Single-Account Focus:** Backend functions operate on single `account_id` strings to ensure clean logic and prevent data crossover between accounts.
- **Stateless UI:** The backend manages data and AI state, while the frontend handles rendering and local UI customizations.

---

## Local Development Setup

### Prerequisites
- Docker Desktop
- Node.js 20+
- C++ Compiler (g++) for the forecasting engine

### 1. Start the Backend and Database
```bash
# Set up your environment variables (Database, TrueLayer API keys, Ollama URL)
cp .env.example .env

# Build and start the Docker containers
docker compose build --no-cache
docker compose up -d
```

### 2. Compile the C++ Engine
```bash
cd services/Forecaster_Agent/mathematics/algorithm
g++ -O3 -shared -fPIC -std=c++17 -o ../hybrid_forecaster.so algorithm.cpp hybrid_algorithm.cpp
```

### 3. Start the Frontend
```bash
cd budai-frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:3000` and the backend API documentation at `http://localhost:8000/docs`.
