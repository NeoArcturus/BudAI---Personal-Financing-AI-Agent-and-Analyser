import joblib
import numpy as np
import re
import pandas as pd


class Categorizer:
    def __init__(self):
        self.rules = {
            "Food & Dining": r"tesco|lidl|asda|greggs|pret|sainsbury|morrisons|aldi|mcdonalds|kfc|starbucks|costa|co-op|vending|chillionrice",
            "Transportation": r"uber|trainline|tfl|bus|rail|stagecoach|bee network|taxi|bolt|national express|beryl",
            "Bills & Utilities": r"lebara|vodafone|ee|o2|british gas|octopus|council tax|water|energy|mobile|circuit laundry",
            "Shopping": r"amazon|ebay|argos|zara|h&m|ikea|currys|adidas|nike|freeprints",
            "Entertainment": r"netflix|spotify|cinema|vue|odeon|steam|playstation|xbox",
            "Health & Wellness": r"boots|pharmacy|gym|nhs|dentist|barber",
            "Transfers & Investments": r"paypal|revolut|transfer|monzo|savings|investment|sent"
        }

    def predict(self, df, embeddings, gbm_model_path, enc_path):
        model = joblib.load(gbm_model_path)
        le = joblib.load(enc_path)

        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        X_infer = np.hstack(
            (embeddings, df['Amount'].values.reshape(-1, 1).astype(np.float32)))

        gbm_predictions = model.predict(X_infer)
        gbm_labels = le.inverse_transform(gbm_predictions)

        final_labels = []
        for i, row in df.iterrows():
            amt = float(row['Amount'])
            text_lower = str(row['Description']).lower()

            if amt > 0.001:
                final_labels.append("Income")
                continue

            matched_cat = None
            for category, pattern in self.rules.items():
                if re.search(pattern, text_lower):
                    matched_cat = category
                    break

            final_labels.append(matched_cat if matched_cat else gbm_labels[i])

        df.loc[:, 'Category'] = final_labels
        return df
