import re
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


class Preprocessor:
    def __init__(self, dataframe):
        self.sentence_tf = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2")
        self.df = dataframe
        self.label_enc = LabelEncoder()
        self.standard_classes = [
            "Income", "Bills & Utilities", "Food & Dining", "Shopping",
            "Transportation", "Entertainment", "Health & Wellness",
            "Travel", "Transfers & Investments", "Other"
        ]
        self.label_enc.fit(self.standard_classes)

        self.category_hints = {
            "Food & Dining": ["supermarket", "grocery", "restaurant", "cafe", "coffee", "tesco", "sainsbury", "lidl", "morrisons", "asda", "aldi", "co-op", "greggs", "pret", "mcdonalds", "pizza", "burger", "dining"],
            "Bills & Utilities": ["mobile", "phone", "bill", "energy", "gas", "electric", "water", "internet", "broadband", "council tax", "utility", "lebara", "vodafone", "ee", "o2"],
            "Transportation": ["uber", "train", "bus", "rail", "tfl", "transport", "ticket", "stagecoach", "bee network", "taxi", "fare"],
            "Entertainment": ["cinema", "movie", "netflix", "spotify", "theatre", "vue", "odeon", "game", "steam", "playstation"],
            "Health & Wellness": ["pharmacy", "doctor", "gym", "boots", "hospital", "wellness", "medicine"],
            "Shopping": ["amazon", "ebay", "clothing", "fashion", "electronics", "retail", "store", "mall", "shoes"],
            "Travel": ["hotel", "airbnb", "flight", "airline", "booking", "travel", "holiday", "staycation"],
            "Transfers & Investments": ["transfer", "savings", "investment", "paypal", "revolut", "splitwise", "sent money", "received money", "sent", "received"],
        }

        self.anchor_embeddings = {
            cat: self.sentence_tf.encode(hints, convert_to_tensor=True)
            for cat, hints in self.category_hints.items()
        }

        self.standard_columns = ["Date", "Amount", "Currency", "Description",
                                 "Payment Reference", "Target Name", "Transaction Type", "Transaction Details Type"]

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r"Card transaction of .* issued by ",
                      "", text, flags=re.IGNORECASE)
        return text.strip()

    def generate_standard_class(self, text):
        text_emb = self.sentence_tf.encode(text, convert_to_tensor=True)
        best_cat = "Other"
        max_sim = -1

        for cat, anchors in self.anchor_embeddings.items():
            sim = util.cos_sim(text_emb, anchors).max().item()
            if sim > max_sim:
                max_sim = sim
                best_cat = cat

        return best_cat if max_sim > 0.4 else "Other"

    def standardize_columns(self):
        manual_map = {"Merchant": "Target Name", "Narrative": "Description",
                      "Reference": "Payment Reference", "Debit Amount": "Amount", "Credit Amount": "Amount"}
        self.df.rename(columns=manual_map, inplace=True)
        df_cols = self.df.columns.tolist()
        df_emb = self.sentence_tf.encode(df_cols, convert_to_tensor=True)
        std_emb = self.sentence_tf.encode(
            self.standard_columns, convert_to_tensor=True)
        cosine_scores = util.cos_sim(df_emb, std_emb)
        all_matches = sorted([{'original': df_cols[i], 'standard': self.standard_columns[j], 'score': cosine_scores[i][j].item()}
                              for i in range(len(df_cols)) for j in range(len(self.standard_columns))], key=lambda x: x['score'], reverse=True)
        used_original, used_standard, rename_map = set(), set(), {}
        for m in all_matches:
            if m['score'] < 0.75 or m['original'] in used_original or m['standard'] in used_standard:
                continue
            rename_map[m['original']] = m['standard']
            used_original.add(m['original'])
            used_standard.add(m['standard'])
        self.df.rename(columns=rename_map, inplace=True)
        self.df = self.df[[
            c for c in self.df.columns if c in self.standard_columns]]

    def preprocess(self):
        self.standardize_columns()
        feature_texts = []
        for _, row in self.df.iterrows():
            target = str(row.get('Target Name', '')).strip()
            desc = self.clean_text(str(row.get('Description', '')))
            feature_texts.append(f"{target} {desc}".strip(
            ) if target.lower() != 'nan' and target != '' else desc)

        self.df["Description"] = feature_texts
        classes = [self.generate_standard_class(t) for t in feature_texts]
        return self.df, self.label_enc, self.label_enc.transform(classes)
