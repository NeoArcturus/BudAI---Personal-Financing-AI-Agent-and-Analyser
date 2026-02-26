import re
import pandas as pd
import xgboost as xgb
import joblib
import numpy as np


class Categorizer:
    def __init__(self):
        self.rules = {
            "Food & Dining": r"tesco|lidl|asda|greggs|pret|sainsbury|morrisons|aldi|mcdonalds|kfc|starbucks|costa",
            "Transportation": r"uber|trainline|tfl|bus|rail|stagecoach|bee network|taxi|bolt",
            "Bills & Utilities": r"lebara|vodafone|ee|o2|british gas|octopus|council tax|water|energy|mobile",
            "Shopping": r"amazon|ebay|argos|zara|h&m|ikea|currys",
            "Entertainment": r"netflix|spotify|cinema|vue|odeon|steam|playstation|xbox",
            "Health & Wellness": r"boots|pharmacy|gym|nhs|dentist",
            "Transfers & Investments": r"paypal|revolut|transfer|monzo|savings|investment"
        }

    def predict(self, df, embeddings, xgb_model_path, enc_path):
        model = xgb.XGBClassifier(n_jobs=1)
        model.load_model(xgb_model_path)
        le = joblib.load(enc_path)

        X_infer = np.hstack(
            (embeddings, df['Amount'].values.reshape(-1, 1).astype(np.float32)))
        xgb_predictions = model.predict(X_infer)
        xgb_labels = le.inverse_transform(xgb_predictions)

        final_labels = []
        for i, (amt, desc) in enumerate(zip(df['Amount'], df['Description'])):
            text_lower = str(desc).lower()
            if amt > 0:
                final_labels.append("Income")
                continue

            matched = False
            for category, pattern in self.rules.items():
                if re.search(pattern, text_lower):
                    final_labels.append(category)
                    matched = True
                    break

            if not matched:
                final_labels.append(xgb_labels[i])

        df.loc[:, 'Category'] = final_labels
        return df
