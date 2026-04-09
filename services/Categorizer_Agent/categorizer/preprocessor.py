import os
import re
import pandas as pd
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class Preprocessor:
    def __init__(self, dataframe, st_model_path):
        self.df = dataframe.copy()
        self.standard_columns = [
            "transaction_id",
            "transaction_uuid",
            "account_id",
            "Date",
            "Amount",
            "Currency",
            "Description",
            "Target Name",
            "Truelayer_classification",
            "Category",
        ]

        self.rules = {
            "Food & Dining": r"(?i)\b(tesco|lidl|asda|greggs|pret|sainsbury|morrisons|aldi|mcdonalds|kfc|starbucks|costa|co-op|vending|chillionrice|subway|grill|restaurant|cafe|pizza|burger|kitchen|food|deliveroo|just eat|uber eats)\b",
            "Transportation": r"(?i)\b(uber|trainline|tfl|transport for london|bus|rail|stagecoach|bee network|taxi|bolt|national express|beryl|transport|coach|train|ticket|avanti|west coast|ctylink)\b",
            "Bills & Utilities": r"(?i)\b(lebara|vodafone|ee|o2|british gas|octopus|council tax|water|energy|mobile|circuit laundry|utility|broadband)\b",
            "Shopping": r"(?i)\b(amazon|prime|ebay|argos|zara|h&m|ikea|currys|adidas|nike|freeprints|shop|store|flexistore)\b",
            "Entertainment": r"(?i)\b(netflix|spotify|cinema|vue|odeon|steam|playstation|xbox|tower bridge|obscura|pub|bar|club|wake|hotstar)\b",
            "Health & Wellness": r"(?i)\b(boots|pharmacy|gym|nhs|dentist|barber|clinic|health|hospital|medical)\b",
            "Transfers & Investments": r"(?i)\b(paypal|revolut|transfer|monzo|savings|investment|sent|added to)\b"
        }

        self.semantic_anchors = {
            "Food & Dining": ["supermarket groceries", "fast food restaurant", "coffee shop cafe", "dining out pub", "vending machine snack", "food delivery takeaway"],
            "Transportation": ["public transport ticket", "taxi ride hail", "train fare rail", "fuel petrol station", "commuter travel", "parking fees"],
            "Bills & Utilities": ["monthly utility bill", "mobile phone contract", "electricity gas water", "internet broadband provider", "insurance premium", "rent payment"],
            "Shopping": ["online retail store", "department store", "clothing apparel fashion", "electronics gadgets", "home furniture household", "general merchandise"],
            "Entertainment": ["streaming subscription service", "cinema movie theater", "video gaming console", "concert music event", "hobby leisure activity", "bookstore"],
            "Health & Wellness": ["pharmacy medicine", "gym membership fitness", "dental medical clinic", "healthcare services", "vitamins supplements", "spa beauty"],
            "Transfers & Investments": ["bank transfer send", "savings account deposit", "stock market investment", "cryptocurrency exchange", "peer to peer payment", "money wire"]
        }

        if not os.path.exists(st_model_path):
            os.makedirs(st_model_path, exist_ok=True)
            SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2").save(st_model_path)

        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.st_model = SentenceTransformer(st_model_path, device=device)

        self.category_names = list(self.semantic_anchors.keys())
        self.category_embeddings = np.zeros((len(self.category_names), 384))

        for i, cat in enumerate(self.category_names):
            anchors = self.semantic_anchors[cat]
            anchor_embs = self.st_model.encode(anchors, convert_to_numpy=True)
            self.category_embeddings[i] = np.mean(anchor_embs, axis=0)

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        return re.sub(r"Card transaction of .* issued by ", "", text, flags=re.IGNORECASE).strip()

    def standardize_columns(self):
        manual_map = {
            "timestamp": "Date", "transaction_date": "Date", "Merchant": "Target Name",
            "Narrative": "Description", "Reference": "Payment Reference",
            "amount": "Amount", "description": "Description", "truelayer_classification": "Truelayer_classification",
            "transaction_uuid": "transaction_uuid", "transaction_id": "transaction_id", "account_id": "account_id"
        }
        self.df.rename(columns=manual_map, inplace=True)
        self.df.columns = [c.capitalize() if c.lower(
        ) in ['amount', 'date'] else c for c in self.df.columns]

        self.df = self.df.loc[:, ~self.df.columns.duplicated()].copy()

        available_cols = [
            c for c in self.standard_columns if c in self.df.columns]
        self.df = self.df[available_cols].copy()

    def _base_preprocess(self):
        self.standardize_columns()
        if "Date" not in self.df.columns:
            self.df.loc[:, "Date"] = pd.to_datetime(
                'today').strftime('%Y-%m-%d')

        feature_texts = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Cleaning"):
            target = str(row.get('Target Name', '')).strip()
            desc = self.clean_text(str(row.get('Description', '')))
            tl_class = str(row.get('Truelayer_classification', '')).strip()

            combined = f"{target} {desc} {tl_class}".strip()
            if combined.lower().startswith('nan '):
                combined = combined[4:]

            feature_texts.append(combined)

        self.df.loc[:, "Description"] = feature_texts
        embeddings = self.st_model.encode(
            self.df["Description"].tolist(), batch_size=32, convert_to_numpy=True)
        return self.df, embeddings

    def preprocess_for_inference(self):
        return self._base_preprocess()

    def preprocess_for_training(self):
        df, embeddings = self._base_preprocess()
        final_labels = []

        for i, row in tqdm(df.iterrows(), total=len(df), desc="Generating Artificial Labels"):
            amt = row['Amount']
            text = row['Description']
            text_lower = text.lower()

            if amt > 0:
                final_labels.append("Income")
                continue

            best_cat = None
            max_sim = -1.0

            window_size = 8
            windows = [text[j:j+window_size]
                       for j in range(len(text) - window_size + 1)]
            if not windows:
                windows = [text]

            window_embs = self.st_model.encode(windows, convert_to_numpy=True)
            sim_matrix = cosine_similarity(
                window_embs, self.category_embeddings)

            best_window_idx, best_cat_idx = np.unravel_index(
                np.argmax(sim_matrix), sim_matrix.shape)
            best_cat = self.category_names[best_cat_idx]

            for category, pattern in self.rules.items():
                if re.search(pattern, text_lower):
                    best_cat = category
                    break

            final_labels.append(best_cat)

        df.loc[:, "Category"] = final_labels
        return df, embeddings
