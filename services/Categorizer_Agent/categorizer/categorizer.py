import joblib
import numpy as np
import re
import pandas as pd


class Categorizer:
    def __init__(self):
        self.rules = {
            "Food & Dining": r"(?i)\b(tesco|lidl|asda|greggs|pret|sainsbury|morrisons|aldi|mcdonalds|kfc|starbucks|costa|co-op|vending|chillionrice|subway|grill|restaurant|cafe|pizza|burger|kitchen|food|deliveroo|just eat|uber eats)\b",
            "Transportation": r"(?i)\b(uber|trainline|tfl|transport for london|bus|rail|stagecoach|bee network|taxi|bolt|national express|beryl|transport|coach|train|ticket|avanti|west coast|ctylink)\b",
            "Bills & Utilities": r"(?i)\b(lebara|vodafone|ee|o2|british gas|octopus|council tax|water|energy|mobile|circuit laundry|utility|broadband)\b",
            "Shopping": r"(?i)\b(amazon|prime|ebay|argos|zara|h&m|ikea|currys|adidas|nike|freeprints|shop|store|flexistore)\b",
            "Entertainment": r"(?i)\b(netflix|spotify|cinema|vue|odeon|steam|playstation|xbox|tower bridge|obscura|pub|bar|club|wake|hotstar)\b",
            "Health & Wellness": r"(?i)\b(boots|pharmacy|gym|nhs|dentist|barber|clinic|health|hospital|medical)\b",
            "Transfers & Investments": r"(?i)\b(paypal|revolut|transfer|monzo|savings|investment|sent|added to)\b"
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
            text_lower = str(row.get('Description', '')).lower()

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
