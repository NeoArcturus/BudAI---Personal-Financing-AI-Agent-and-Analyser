# BudAI: Personal Finance AI Agent and Analyser

BudAI is a production-grade transaction categorization engine designed to effectively bridge the gap between semantic text understanding and numerical financial logic. By moving beyond text-only models, the system ensures that financial context (merchant intent) and mathematical reality (transaction magnitude) are processed in parallel to solve the "Math Blindness" of standard Language Models.



## Project Overview

Standard NLP models often fail to distinguish between Income and Expenses due to numerical magnitude blindness (treating -50.00 and +50.00 as similar semantic strings). BudAI utilizes a hybrid architecture combining **Sentence-Transformer Embeddings** with **XGBoost** to achieve 98% weighted F1-accuracy.

### Core Features & Engineering Highlights

* **Hybrid Machine Learning Architecture:** Engineered a multi-stage classification pipeline combining **Sentence-Transformer Embeddings** (all-MiniLM-L6-v2) for semantic context with **XGBoost** for numerical feature processing.
* **Feature Engineering & Data Distillation:** Designed a custom preprocessing suite to handle messy bank CSV data. Implemented fuzzy semantic column mapping and regex-based boilerplate removal to isolate merchant intent from generic noise (e.g., "Card transaction issued by...").
* **Solving the Accuracy Paradox:** Addressed class imbalance by implementing synthetic oversampling and class-weighting strategies, improving the **Macro F1-score from 0.37 to 0.85**.
* **Semantic Fallback Logic:** Integrated a robust post-prediction layer using **Cosine Similarity** against high-confidence category anchors. This ensures prominent merchants (Tesco, Uber, Lebara) are categorized with near-perfect accuracy regardless of description variations.



## Tech Stack
* **Language:** Python 3.11+
* **Modeling:** XGBoost (Gradient Boosting), Transformers (MiniLM-L6-v2)
* **NLP Tools:** Sentence-Transformers, HuggingFace, Regex-based distillation
* **Data Science:** Scikit-Learn (Stratified Splits, Oversampling), Pandas, NumPy

## Project Structure

* `preprocessor.py`: Standardizes messy bank CSVs and handles column mapping via fuzzy semantic matching.
* `model_trainer.py`: Encodes descriptions into 384-dimensional vectors, stacks them with numerical amounts, and trains the XGBoost classifier.
* `categorizer.py`: The production-ready inference engine featuring strict directional logic to prevent spending from being categorized as income.
* `xgb_model.json`: The serialized gradient boosting model.
* `label_encoder.joblib`: Mapping of integer predictions to human-readable categories.

## Performance Metrics

| Category | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Transfers & Investments** | 1.00 | 1.00 | 1.00 |
| **Food & Dining** | 0.95 | 1.00 | 0.98 |
| **Transportation** | 1.00 | 1.00 | 1.00 |
| **Bills & Utilities** | 0.75 | 1.00 | 0.86 |
| **Weighted Average** | **0.99** | **0.98** | **0.98** |

## Installation and Usage

### 1. Setup
```bash
pip install xgboost sentence-transformers pandas scikit-learn joblib tqdm