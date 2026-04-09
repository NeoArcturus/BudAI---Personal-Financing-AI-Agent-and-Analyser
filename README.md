# BudAI - Personal Financing AI Agent and Analyzer

BudAI is a full-stack personal finance assistant that combines:
- a `FastAPI` backend for auth, account sync, chat, and tool execution,
- a `Next.js` frontend (`budai-frontend`) for dashboard + visualizations,
- local LLM orchestration (`LangChain` + `Ollama`) for conversational financial analysis,
- and Python/C++ forecasting + analytics services backed by SQLite.

## Current Architecture

### Backend (active API)
- Primary entrypoint: `main.py` (FastAPI, port `8080`).
- Database: SQLite by default (`budai_memory.db`) via SQLAlchemy models.
- Token refresh job: APScheduler runs every 45 minutes to refresh TrueLayer access tokens.
- Auth model: bearer token currently uses `user_uuid` returned by login.

### Frontend
- Location: `budai-frontend/`
- Stack: Next.js 16, React 19, Tailwind 4, Chart.js.
- Expected backend base URL: `http://localhost:8080`.

### Legacy server
- `server.py` (Flask) still exists but the implementation is now FastAPI-first.

## Implemented Capabilities

- TrueLayer connection flow (auth link, callback token exchange, refresh, revoke).
- User login/bootstrap with local user storage.
- Account discovery, account transaction retrieval, and provider revocation.
- Streaming AI chat endpoint backed by local Ollama model (`qwen3:4b` by default).
- Tool execution endpoint for analytics/forecasting payloads:
  - classification summaries,
  - category spend aggregation,
  - historical expense timelines,
  - cash-flow analysis,
  - expense forecasting,
  - balance forecasting,
  - health radar and financial health metrics.
- Multi-account resolution (`ALL`, bank name(s), or account id).
- Chart payload caching via `chart_cache` table (`CACHE_*` ids).

## Project Structure (Current)

```text
BudAI---Personal-Financing-AI-Agent-and-Analyser/
├── main.py
├── config.py
├── requirements.txt
├── models/
│   └── database_models.py
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
│   ├── budai_chat_service.py
│   ├── tools.py
│   ├── api_integrator/
│   ├── Categorizer_Agent/
│   ├── Forecaster_Agent/
│   └── Analyser_Agent/
└── budai-frontend/
    ├── package.json
    └── ...
```

## Backend Setup

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment variables

Create a `.env` file in the project root:

```env
# Database / auth
DATABASE_URL=sqlite:///budai_memory.db
SECRET_KEY=replace-this-secret

# TrueLayer
BASE_URL=https://api.truelayer.com/data/v1
AUTH_LINK_URL=https://auth.truelayer.com
CLIENT_ID=your_truelayer_client_id
CLIENT_SECRET=your_truelayer_client_secret
REDIRECT_URI=http://localhost:8080/callback

# Encryption key (Fernet key)
ENCRYPTION_KEY=your_fernet_key
```

### 4) Run backend API

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

## Frontend Setup

```bash
cd budai-frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:3000` by default.

## API Endpoints (Current)

### Auth
- `POST /api/auth/login`
- `GET /api/auth/truelayer/status`
- `GET /callback`
- `POST /api/auth/connections/extend`
- `POST /api/auth/connections/revoke`

### Accounts
- `GET /api/accounts/`
- `GET /api/accounts/{account_id}/transactions`
- `DELETE /api/accounts/{provider_id}`

### Chat
- `POST /api/chat/` (streaming text response)

### Media / Analytics execution
- `POST /api/media/execute`

`tool_name` currently supports:
- `plot_cash_flow_mixed`
- `classify_financial_data`
- `create_bargraph_chart_and_save`
- `find_total_spent_for_given_category`
- `plot_health_radar`
- `plot_expenses`
- `generate_expense_forecast`
- `generate_financial_forecast`

## Notes

- The chat agent is configured for local Ollama at `http://localhost:11434`; make sure Ollama is running with the expected model.
- The repository includes both FastAPI and Flask server files; use `main.py` unless you intentionally need the legacy Flask path.
- Some advanced forecasting logic depends on native C++ components under `services/Forecaster_Agent/mathematics/algorithm/`.
