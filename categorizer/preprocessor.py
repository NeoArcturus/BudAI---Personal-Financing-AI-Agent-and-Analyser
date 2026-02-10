from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


class Preprocessor:
    def __init__(self, data_file):
        self.sentence_tf = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2")
        self.df = pd.read_csv(data_file)
        self.label_enc = LabelEncoder()

        self.standard_classes = [
            "Income", "Bills & Utilities", "Food & Dining", "Shopping",
            "Transportation", "Entertainment", "Health & Wellness",
            "Travel", "Transfers & Investments", "Other"
        ]
        self.encoded_classes = self.sentence_tf.encode(
            self.standard_classes, convert_to_tensor=True)

        self.standard_columns = [
            "Date", "Amount", "Currency", "Description",
            "Payment Reference", "Target Name", "Transaction Type", "Transaction Details Type"
        ]

    def generate_standard_class(self, text):
        em2 = self.sentence_tf.encode(text, convert_to_tensor=True)
        best_idx = util.cos_sim(em2, self.encoded_classes).argmax().item()
        return self.standard_classes[best_idx]

    def standardize_columns(self):
        manual_map = {
            "Merchant": "Target Name",
            "Narrative": "Description",
            "Reference": "Payment Reference",
            "Debit Amount": "Amount",
            "Credit Amount": "Amount"
        }
        self.df.rename(columns=manual_map, inplace=True)

        df_cols = self.df.columns.tolist()
        df_emb = self.sentence_tf.encode(df_cols, convert_to_tensor=True)
        std_emb = self.sentence_tf.encode(
            self.standard_columns, convert_to_tensor=True)

        cosine_scores = util.cos_sim(df_emb, std_emb)

        all_matches = []
        for i in range(len(df_cols)):
            for j in range(len(self.standard_columns)):
                all_matches.append({
                    'original': df_cols[i],
                    'standard': self.standard_columns[j],
                    'score': cosine_scores[i][j].item()
                })

        all_matches = sorted(
            all_matches, key=lambda x: x['score'], reverse=True)

        used_original = set()
        used_standard = set()
        rename_map = {}

        for match in all_matches:
            if match['score'] < 0.75:
                continue

            if match['original'] in used_original or match['standard'] in used_standard:
                continue

            rename_map[match['original']] = match['standard']
            used_original.add(match['original'])
            used_standard.add(match['standard'])

        self.df.rename(columns=rename_map, inplace=True)

        final_cols = [c for c in self.df.columns if c in self.standard_columns]
        self.df = self.df[final_cols]

    def preprocess(self):
        self.standardize_columns()

        required = ["Date", "Amount"]
        missing = [c for c in required if c not in self.df.columns]

        if missing:
            raise ValueError(f"Missing essential fields: {missing}")

        if "Description" not in self.df.columns:
            if "Target Name" in self.df.columns:
                self.df["Description"] = self.df["Target Name"]
            else:
                self.df["Description"] = "Unknown Transaction"

        descriptions = self.df["Description"]
        classes = [self.generate_standard_class(
            str(desc)) for desc in descriptions]
        label = self.label_enc.fit_transform(classes)

        return self.df, self.label_enc, label
