# BudAI - Personal Financing AI Agent and Analyzer

## Project Description

BudAI is a comprehensive, full-stack personal finance assistant designed to act as an empathetic, highly capable financial advisor. By securely integrating with users' bank accounts via TrueLayer, BudAI ingests and processes transactional data to provide actionable insights, budget categorization, and predictive forecasting. At its core, the system relies on a robust FastAPI backend and an advanced local LLM orchestration layer powered by LangChain, LangGraph, and Ollama. This architecture allows users to interact with their financial data through natural language, asking complex questions about past spending trends or future wealth velocity, and receiving highly accurate, context-aware responses.

Technical reliability and user experience are central to BudAI's design, highlighted by its recent migration to a multi-agent node orchestration framework. This state-graph approach drastically reduces LLM hallucination risks by cleanly separating intent routing, specialized analytical workers (such as categorizers and forecasters), and persona-driven response generation. Furthermore, the system seamlessly bridges the gap between text and UI by using deterministic backend logic to trigger interactive visualizations—like cash-flow diagrams, expense charts, and financial health radars—directly within the Next.js frontend. The result is a secure, responsive, and deeply interactive dashboard that transforms raw banking data into a personalized financial roadmap.

## Tech Stack

BudAI is a full-stack personal finance assistant that combines:

- a `FastAPI` backend for auth, account sync, chat, and tool execution,
- a `Next.js` frontend (`budai-frontend`) for dashboard + visualizations,
- local multi-agent LLM orchestration (`LangChain`, `LangGraph`, and `Ollama`) for conversational financial analysis,
- and `Python/C++ forecasting + analytics` services backed by `SQLite`.

## Current Architecture

### Backend (active API)

- Primary entrypoint: `main.py` (FastAPI, port `8080`).
- Database: SQLite by default (`budai_memory.db`) via SQLAlchemy models.
- Token refresh job: APScheduler runs every 45 minutes to refresh TrueLayer access tokens.
- Auth model: bearer token currently uses `user_uuid` returned by login.
- AI Orchestration: Multi-agent state graph using `LangGraph`. Includes an Intent Router, specialized worker nodes, a programmatic UI tagger, and a dedicated Persona response generator.

### Frontend

- Location: `budai-frontend/`
- Stack: Next.js 16, React 19, Tailwind 4, Chart.js.
- Expected backend base URL: `http://localhost:8080`.
- Read more on the frontend in the `budai-frontend/README.md` file for more details.

## Implemented Capabilities

- TrueLayer connection flow _`(auth link, callback token exchange, refresh, revoke)`_.
- User login/bootstrap with `local user storage`.
- Account discovery, account transaction retrieval, and provider revocation.
- Streaming AI chat endpoint backed by a modular state graph and local Ollama models.
- Multi-agent orchestration for **_`robust intent routing`_** and **_`hallucination-resistant`_** tool execution.
- Deterministic UI trigger generation to bypass LLM formatting unreliability.
- Tool execution endpoint for analytics/forecasting payloads:
  - `classification summaries`,
  - `category spend aggregation`,
  - `historical expense timelines`,
  - `cash-flow analysis`,
  - `expense forecasting`,
  - `balance forecasting`,
  - `health radar` and `financial health metrics`.
- Multi-account resolution (`ALL`, bank name(s), or account id).
- Chart payload caching via `chart_cache` table (`CACHE_*` ids).

## Project Structure (Current)

```text
BudAI---Personal-Financing-AI-Agent-and-Analyser/
├── main.py
├── config.py
├── requirements.txt
├── models/
│   ├── database_models.py
│   └── graph_state.py
├── controllers/
│   ├── auth_controller.py
│   ├── account_controller.py
│   ├── chat_controller.py
│   └── media_controller.py
├── middleware/
│   └── auth_middleware.py
├── schemas/
│   └── api_schema.py
├── services/
│   ├── orchestrator_graph.py
│   ├── tools.py
│   ├── api_integrator/
│   ├── workers/
│   │   ├── analyser_worker.py
│   │   ├── forecaster_worker.py
│   │   ├── categorizer_worker.py
│   │   └── health_worker.py
│   ├── Categorizer_Agent/
│   ├── Forecaster_Agent/
│   └── Analyser_Agent/
└── budai-frontend/
    ├── package.json
    └── ...
```

---

## Backend Setup

`1. Create and activate a virtual environment`

```
python -m venv .venv
source .venv/bin/activate
```

`2. Install dependencies`

```
pip install -r requirements.txt
```

`3. Configure environment variables`

Create a .env file in the project root:

```
DATABASE_URL=sqlite:///budai_memory.db
SECRET_KEY=replace-this-secret

BASE_URL=[https://api.truelayer.com/data/v1]
AUTH_LINK_URL=[https://auth.truelayer.com]
CLIENT_ID=your_truelayer_client_id
CLIENT_SECRET=your_truelayer_client_secret
REDIRECT_URI=http://localhost:8080/callback
```

ENCRYPTION_KEY=your_fernet_key

`4. Run backend API`

```
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

`5. Frontend Setup`

```
cd budai-frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:3000` by default.

---

## API Endpoints (Current)

Auth

```
|-POST /api/auth/login
|-GET /api/auth/truelayer/status
|-GET /callback
|-POST /api/auth/connections/extend
|-POST /api/auth/connections/revoke
```

Accounts

```
|-GET /api/accounts/
|-GET /api/accounts/{account_id}/transactions
|-DELETE /api/accounts/{provider_id}
```

Chat

```
POST /api/chat/ (streaming text response)
```

Media / Analytics execution

```
POST /api/media/execute
```

tool_name currently supports:

```
|-plot_cash_flow_mixed
|-classify_financial_data
|-create_bargraph_chart_and_save
|-find_total_spent_for_given_category
|-plot_health_radar
|-plot_expenses
|-generate_expense_forecast
|-generate_financial_forecast
```

---

## Notes

- The orchestration graph is configured for local Ollama at `http://localhost:11434`. Ensure the required models are pulled and running (e.g., qwen3:4b for intent routing and qwen3.5:4b for persona generation).

- Some advanced forecasting logic depends on native C++ components under services/Forecaster_Agent/mathematics/algorithm/.
