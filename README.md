# BudAI: Full-Stack Personal Finance Intelligence

BudAI is an autonomous, privacy-first financial intelligence system designed to standardize, categorize, and forecast personal expenditures. Evolving from a terminal script into a full-stack web application, BudAI bridges a modern React/Next.js dashboard with a high-performance Flask backend powered by C++ engineering, advanced Python machine learning, and local Large Language Models (LLMs).

The system operates as a **Hybrid Agentic Ecosystem**, utilizing a strict, stoic conversational AI orchestrator to trigger highly specialized, independent background engines while maintaining strict multi-account data isolation.

BudAI is structured as a modular pipeline that separates user interaction, API orchestration, financial intelligence agents, high-performance computation, and persistent storage. The frontend communicates with a lightweight Flask gateway that validates requests, resolves banking contexts, and routes structured instructions to internal AI tools. These tools coordinate specialized agents responsible for categorization, forecasting, and financial health analysis. Heavy numerical simulations are delegated to a compiled C++ engine, while structured financial data and authentication states are maintained in a secure SQLite persistence layer. External telemetry such as banking synchronization and language model inference integrates seamlessly into this pipeline while preserving a privacy-first, locally executed architecture.

---

# Agentic Architecture & The Frontend

BudAI operates through a central conversational router (BudAI Chat) that delegates complex financial tasks to specialized tools.

## The Agentic UI (Next.js)

The frontend is a dynamic Next.js dashboard that visualizes the AI's outputs. It features a "Smart Sync" chat interface that automatically detects which bank account (e.g., Revolut, Wise) the user is viewing or discussing, passing this context seamlessly to the backend API to prevent data bleed.

The interface also renders visual analytics including financial charts, transaction distributions, and forecasting outputs produced by backend agents.

---

# Core Backend Components (Flask & Python)

## Orchestrator Agent (BudAI Chat)

Powered by LangChain's AgentExecutor and running locally via Ollama (optimized for models like Llama 3.1 8B or Qwen 2.5 7B). This agent acts as a grounded financial advisor.

It relies on highly strict prompt engineering to prevent "thought leaking" and JSON hallucination, ensuring it only delivers accurate, data-driven insights without emotional fluff or emojis.

The orchestrator acts primarily as a **task router**, selecting specialized tools rather than performing heavy computation itself.

## Independent TrueLayer Resolution

Agents no longer rely on the orchestrator for authentication.

The `UserAccount` class allows each sub-agent to autonomously:

- Securely query the SQLite database
- Resolve natural language bank names (like "Monzo") into exact `account_id`s
- Connect to the TrueLayer API independently

This design prevents cross-account data leakage and ensures strict isolation between financial contexts.

## Categorization Agent

Acts as the sensory organ of the system. It:

- Fetches raw bank transaction data
- Standardizes financial records
- Performs a two-phase categorization process utilizing a local XGBoost classifier

This agent converts unstructured transaction descriptions into structured financial categories.

## Forecaster Agent (C++/Python)

The forecasting engine models financial behavior using stochastic simulations.

It:

- Calculates drift and volatility from real balances
- Uses a custom C++ engine to run Monte Carlo simulations
- Generates visual convergence charts
- Produces narrative financial forecasts

These forecasts estimate potential financial trajectories under multiple behavioral scenarios.

---

# Directory Structure

The project maintains strict modularity, separating the Next.js frontend, Flask API, machine learning environments, and localized media storage.

```
BudAI/
├── frontend/                     # Next.js 16 (App Router) Dashboard
│   ├── app/dashboard/            # Agentic UI, Transaction Feeds, and Chart Rendering
│   └── components/               # Reusable UI elements (Tailwind CSS, Lucide Icons)
├── backend/                      # Python Flask Server (Port 8080)
│   ├── api_integrator/           # Autonomous TrueLayer data fetching and token resolution
│   ├── Categorizer_Agent/        # XGBoost model trainer and preprocessor
│   ├── Forecaster_Agent/         # C++ Monte Carlo SDE Simulation Engine
│   ├── Analyser_Agent/           # Expense and Category Distribution plotting
│   ├── saved_media/              # Dynamically isolated outputs (Images & CSVs tagged by account_id)
│   ├── budai_memory.db           # SQLite database for tokens, caching, and embeddings
│   ├── server.py                 # Flask API routing (Chat, Accounts, Media Serving)
│   └── tools.py                  # LangChain tool definitions (Data strictly separated by account_id)
```

---

# Key Technical Features

## Multi-Account Data Isolation

To support users with multiple bank accounts (e.g., Wise, Revolut), the backend dynamically tags every generated CSV, classification report, and Matplotlib chart with its specific `account_id` (e.g., `categorized_data_14748516.csv`).

The Next.js frontend fetches these specific files via the Flask `/api/media/` routes, ensuring dashboards never bleed data between accounts.

---

## Hybrid Local AI Orchestration

BudAI is entirely privacy-first. It relies on a local LLM for natural language processing and task routing, ensuring that sensitive financial questions and data are never sent to external providers like OpenAI.

The system uses strict positive-constraint prompting to enforce deterministic tool execution.

---

## The Hybrid XGBoost Classifier

BudAI moves beyond simple keyword matching by employing a hybrid NLP + Gradient Boosting classification pipeline to achieve high-precision transaction labeling.

### Algorithm

- XGBoost Classifier (`multi:softprob` objective for multi-class probability scoring)

### Zero-Shot Bootstrapping

On the first run, a deterministic regex rule-engine labels the data to bootstrap the ML model, allowing it to generate artificial training targets without manual intervention.

### Feature Engineering

The model does not train on raw text. Instead, the feature matrix (`X`) is formed by concatenating:

- Semantic Embeddings: High-dimensional dense vectors generated by Sentence-Transformers (`all-MiniLM-L6-v2`)
- Transaction Amount: The numerical value (`Amount`), ensuring the model understands magnitude and direction (spend vs. income)

---

## Stochastic Habit Modeling (C++)

To overcome the Python Global Interpreter Lock (GIL) and process heavy mathematical computations instantly, BudAI offloads forecasting to a compiled C++ engine.

It:

- Calculates historical drift (μ) and volatility (σ) from real banking data
- Runs hundreds of thousands of parallel Monte Carlo simulation paths
- Predicts "Careless", "Expected", and "Optimal" balance trajectories
- Generates Expense Convergence charts to track expected outgoing capital against historical baselines

---

# Technical Stack

## Frontend

- React 18
- Next.js 16 (Turbopack)
- Tailwind CSS
- Lucide React

## Backend API

- Python 3.9+
- Flask
- Flask-CORS

## Core Engine Languages

- Python 3.9+
- C++ (Standard 17+)

## AI / ML

- PyTorch
- Sentence-Transformers
- XGBoost
- Scikit-learn

## Orchestration

- LangChain
- Ollama (Local LLMs)

## Data & Storage

- Pandas
- SQLite
- DiskCache
