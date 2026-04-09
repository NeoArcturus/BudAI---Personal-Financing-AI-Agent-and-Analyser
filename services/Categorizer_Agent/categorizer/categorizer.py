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
        gbm_proba = model.predict_proba(X_infer)
        gbm_conf = np.max(gbm_proba, axis=1)

        final_labels = []
        final_confidences = []
        model_labels = []
        needs_review_threshold = 0.62
        income_keywords = ("salary", "payroll", "wage", "interest", "bonus")
        transfer_like_keywords = ("refund", "reversal", "chargeback", "transfer", "cashback")

        for i, row in df.iterrows():
            amt = float(row['Amount'])
            text_lower = str(row.get('Description', '')).lower()
            model_label = gbm_labels[i]
            model_conf = float(gbm_conf[i])
            model_labels.append(model_label)

            if amt > 0.001 and any(k in text_lower for k in income_keywords) and not any(k in text_lower for k in transfer_like_keywords):
                final_labels.append("Income")
                final_confidences.append(max(model_conf, 0.92))
                continue

            matched_cat = None
            for category, pattern in self.rules.items():
                if re.search(pattern, text_lower):
                    matched_cat = category
                    break

            if matched_cat:
                final_labels.append(matched_cat)
                final_confidences.append(max(model_conf, 0.95))
            elif model_conf < needs_review_threshold:
                final_labels.append("Needs Review")
                final_confidences.append(model_conf)
            else:
                final_labels.append(model_label)
                final_confidences.append(model_conf)

        df.loc[:, 'Category'] = final_labels
        df.loc[:, 'Model_Prediction'] = model_labels
        df.loc[:, 'Confidence'] = final_confidences
        return df
