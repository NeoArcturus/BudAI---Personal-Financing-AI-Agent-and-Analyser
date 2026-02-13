# BudAI: Personal Financing AI Agent and Analyser

BudAI is an autonomous financial intelligence system designed to standardize, categorize, and forecast personal expenditures. By bridging the gap between high-performance engineering (C++) and advanced machine learning (Python), BudAI provides users with a unified financial view and proactive, behaviour-aware recommendations.

---

## Core Architecture

The system utilizes a hybrid neural-symbolic approach to manage personal finances:

### Neutralization

A hardware-accelerated Preprocessor that standardizes messy bank statements from multiple institutions using a hybrid Sentence Transformer (MiniLM and MPNet).

### Intelligence

A transaction classifier combining semantic embeddings with XGBoost to achieve high-precision categorization.

### Forecasting

A high-performance C++ engine that models user habits via the Ornstein-Uhlenbeck (OU) process and market volatility via Geometric Brownian Motion (GBM).

### Agentic AI

A RAG-enabled (Retrieval-Augmented Generation) interface that allows users to query their financial data and receive personalized recommendations.

---

## Key Features

### Hybrid Transaction Classification

BudAI uses a multi-stage NLP pipeline to ensure speed and accuracy.

The system utilizes:

- `all-MiniLM-L12-v2` for rapid column standardization
- `all-mpnet-base-v2` for deep semantic classification of transaction descriptions

It features hardware-agnostic acceleration for:

- Apple Silicon (MPS)
- NVIDIA GPUs (CUDA)

---

### Stochastic Expenditure Forecasting

Unlike traditional budget trackers, BudAI implements a dual-model forecasting layer in C++.

#### Ornstein-Uhlenbeck Model

Calibrates to user spending habits by calculating:

- Mean reversion speed
- Habitual mean spending levels

#### Geometric Brownian Motion

Simulates global commodity paths (e.g., crude oil) to predict the impact of external inflation on user expenses.

---

### Commodity-Linked Hedging

The system identifies the **Beta correlation** between user categories (like Transportation) and commodity prices.

It suggests **Hedged Paths**, such as:

- Pre-buying
- Bulk purchasing

These recommendations help minimize expenditure when market forecasts indicate price increases.

---

### Agentic RAG Interface

Forecasted paths, behavioral patterns, and market sensitivities are stored in a vector database.

The AI agent uses Retrieval-Augmented Generation (RAG) to provide grounded responses to queries like:

> "How can I optimize my spending for next month?"

---

## Technical Stack

### Machine Learning

- Python
- PyTorch
- Sentence-Transformers
- XGBoost
- Scikit-learn

### Performance Engineering

- C++ (SDE Simulation Engine)

### Data Processing

- Pandas
- NumPy
- Joblib

### Hardware Acceleration

- MPS (Apple Silicon)
- CUDA (NVIDIA)

---

## Getting Started

### Prerequisites

- Python 3.9+
- C++ compiler (g++ or clang)
- PyTorch with MPS or CUDA support

---

## Installation

Clone the repository:

```bash
git clone <repo-url>
cd budai
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Compile the C++ forecasting engine:

```bash
g++ -shared -fPIC -O3 -o forecasting/libforecaster.so forecasting/Forecaster.cpp
```

---

## Usage

### Training

Train the categorizer using historical financial data:

```bash
python main.py --train --file your_bank_statement.csv
```

---

### Forecasting and Analysis

Run the autonomous analyzer:

```bash
python main.py --file your_bank_statement.csv
```

The system will:

- Categorize transactions
- Run 5,000+ stochastic simulations per category
- Prepare the Agentic AI for natural language queries

---

## Vision

BudAI aims to evolve from a financial tracker into a **personal financial intelligence system** capable of understanding behavioral spending patterns, market dynamics, and optimization strategies in a unified framework.
