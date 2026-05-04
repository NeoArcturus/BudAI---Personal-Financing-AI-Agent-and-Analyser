import os
import json
import joblib
import numpy as np
import re
import pandas as pd


class Categorizer:

    def __init__(self):
        rules_path = os.path.join(os.path.dirname(
            __file__), "..", "budai_category_rules.json")
        with open(rules_path, "r") as f:
            rules_data = json.load(f)

        self.rules = rules_data["rules"]
        self.scam_fraud_keywords = self.rules.get("High-Risk / Anomaly", "")

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
        transfer_like_keywords = (
            "refund", "reversal", "chargeback", "transfer", "cashback")

        expenses = df[df['Amount'] < 0]['Amount'].abs()
        if not expenses.empty and len(expenses) > 5:
            mean_exp = expenses.mean()
            std_exp = expenses.std()
            anomaly_threshold = max(mean_exp + (3 * std_exp), 500.0)
        else:
            anomaly_threshold = float('inf')

        for i, row in df.iterrows():
            amt = float(row['Amount'])
            abs_amt = abs(amt)
            text_lower = str(row.get('Description', '')).lower()
            model_label = gbm_labels[i]
            model_conf = float(gbm_conf[i])
            model_labels.append(model_label)

            is_scam_keyword = re.search(
                self.scam_fraud_keywords, text_lower) if self.scam_fraud_keywords else False
            is_statistical_anomaly = (amt < 0) and (
                abs_amt >= anomaly_threshold)

            if is_scam_keyword or is_statistical_anomaly:
                final_labels.append("High-Risk / Anomaly")
                final_confidences.append(0.99)
                continue

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
