# BudAI: Personal Financing AI Agent and Analyser

BudAI is an autonomous financial intelligence system designed to standardize, categorize, and forecast personal expenditures. By bridging high-performance C++ engineering with advanced Python machine learning, BudAI provides a unified, proactive financial view.

The system is evolving from a single script into a **Multi-Agent Ecosystem** where specialized agents collaborate via shared memory to provide deep financial insights.

---

## Agentic Architecture

BudAI operates through a decentralized constellation of autonomous agents, each with a defined role and its own `main.py` entry point. They interact through a **Shared Memory Substrate** comprising a Vector Database (ChromaDB) for long-term semantic history and a Disk Cache for high-speed local state.

### Current & Future Agents

- **Categorization Agent (Python - _Current_):** Acts as the sensory organ. It fetches raw bank data via TrueLayer, standardizes it, and performs rule-based categorization. It generates embeddings **post-categorization** to index "Ground Truth" into the Vector DB.
- **Forecaster Agent (C++/Python - _In Development_):** The mathematical engine. It uses C++ to run 5,000+ stochastic simulations (Ornstein-Uhlenbeck and GBM) to predict habit-based expenditure and commodity-linked risks.
- **News Reader Agent (Python - _Planned_):** Scans geo-political financial news to generate a "Sentiment Vector." This enhances commodity forecasting by adjusting for market volatility based on real-world events.
- **Analyser Agent (Python - _Planned_):** The visualization layer. It synthesizes historical data from the Perception Agent and projections from the Forecaster to display comprehensive risk and gap analyses.
- **User Input Agent (LLM-based - _Planned_):** A natural language interface (Voice/Text) that queries the shared Vector DB to answer complex user questions like, _"Can I buy this based on my current forecast?"_

---

## Directory Structure

To maintain modularity, each agent resides in its own top-level directory as a peer service.

```text
BudAI/
├── agent_vector_db/              # Shared Long-term Memory (ChromaDB)
├── agent_cache/                  # Shared Short-term Memory (DiskCache)
├── Categorization Agent/         # Entry: main.py (API fetch & Categorization)
├── Forecaster Agent/             # Entry: main.py (C++ SDE Simulation Engine)
├── Analyser Agent/               # Entry: main.py (UI & Dashboard)
├── Market Agent/                 # Entry: main.py (Commodity Market Reader)
└── News_Reader Agent/            # Entry: main.py (Sentiment & Geo-Politics)
```

## Key Technical Features

### Semantic Memory Substrate

Unlike standard trackers, BudAI uses **Vector Similarity** to preserve continuity across sessions.

- **Substrate:** ChromaDB for semantic search and reliable shared state.
- **Interface:** Standardized Python `Preprocessor` ensuring no `NaN` data reaches the "brain".

### Stochastic Habit Modeling (C++) (_Under development_)

A high-performance C++ engine models user habits via the **Ornstein-Uhlenbeck (OU) process**, calculating:

- **Mean reversion speed**: How quickly you return to average spending levels.
- **Habitual mean**: Your baseline "equilibrium" expenditure.

### Commodity-Linked Hedging (_Under development_)

BudAI identifies **Beta correlation** between user categories (e.g., Transportation) and global commodity paths (e.g., Crude Oil) simulated via **Geometric Brownian Motion (GBM)**.

### Hardware Acceleration & ML Stack

- **Transformers:** Utilizes `Sentence-Transformers` for post-categorization embeddings.
- **Acceleration:** Hardware-agnostic support for Apple Silicon (MPS) and NVIDIA GPUs (CUDA).

### Technical Stack

- Languages: Python 3.9+, C++ (Standard 17+)
- AI/ML: PyTorch, Sentence-Transformers, XGBoost, Scikit-learn
- Memory/RAG: ChromaDB, DiskCache
- Acceleration: Apple Silicon (MPS), NVIDIA GPUs (CUDA)
