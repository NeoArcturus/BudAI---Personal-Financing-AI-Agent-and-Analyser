import os
import re
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class Preprocessor:
    def __init__(self, dataframe, st_model_path):
        self.df = dataframe.copy()
        self.standard_columns = ["Date", "Amount",
                                 "Currency", "Description", "Target Name"]
        self.rules = {
            "Food & Dining": r"tesco|lidl|asda|greggs|pret|sainsbury|morrisons|aldi|mcdonalds|kfc|starbucks|costa",
            "Transportation": r"uber|trainline|tfl|bus|rail|stagecoach|bee network|taxi|bolt",
            "Bills & Utilities": r"lebara|vodafone|ee|o2|british gas|octopus|council tax|water|energy|mobile",
            "Shopping": r"amazon|ebay|argos|zara|h&m|ikea|currys",
            "Entertainment": r"netflix|spotify|cinema|vue|odeon|steam|playstation|xbox",
            "Health & Wellness": r"boots|pharmacy|gym|nhs|dentist",
            "Transfers & Investments": r"paypal|revolut|transfer|monzo|savings|investment"
        }

        if not os.path.exists(st_model_path):
            os.makedirs(st_model_path, exist_ok=True)
            SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2").save(st_model_path)
        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.st_model = SentenceTransformer(st_model_path, device=device)

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        return re.sub(r"Card transaction of .* issued by ", "", text, flags=re.IGNORECASE).strip()

    def standardize_columns(self):
        manual_map = {
            "timestamp": "Date",
            "transaction_date": "Date",
            "Merchant": "Target Name",
            "Narrative": "Description",
            "Reference": "Payment Reference",
            "amount": "Amount",
            "description": "Description"
        }
        self.df.rename(columns=manual_map, inplace=True)
        self.df.columns = [c.capitalize() if c.lower(
        ) in ['amount', 'date'] else c for c in self.df.columns]
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
            feature_texts.append(f"{target} {desc}".strip(
            ) if target.lower() != 'nan' and target != '' else desc)
        self.df.loc[:, "Description"] = feature_texts

        embeddings = self.st_model.encode(
            self.df["Description"].tolist(), batch_size=32, convert_to_numpy=True)
        return self.df, embeddings

    def preprocess_for_inference(self):
        return self._base_preprocess()

    def preprocess_for_training(self):
        df, embeddings = self._base_preprocess()
        final_labels = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating Artificial Labels"):
            text_lower = row['Description'].lower()
            amt = row['Amount']
            assigned = False
            if amt > 0:
                final_labels.append("Income")
                continue
            for category, pattern in self.rules.items():
                if re.search(pattern, text_lower):
                    final_labels.append(category)
                    assigned = True
                    break
            if not assigned:
                final_labels.append("Other")
        df.loc[:, "Category"] = final_labels
        return df, embeddings
