import re
import pandas as pd
from tqdm import tqdm


class Preprocessor:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.standard_columns = ["Date", "Amount",
                                 "Currency", "Description", "Target Name"]

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

    def preprocess(self):
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
        return self.df
