import xgboost as xgb
import joblib
import numpy as np
import re
from sentence_transformers import SentenceTransformer


class Categorizer:
    def __init__(self, model_file_path, label_enc_path):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_file_path)
        self.label_enc = joblib.load(label_enc_path)
        self.semantic_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2")

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r"Card transaction of .* issued by ",
                      "", text, flags=re.IGNORECASE)
        return text.strip()

    def categorize_data(self, df):
        feature_texts = []
        for _, row in df.iterrows():
            target = str(row.get('Target Name', '')).strip()
            desc = self.clean_text(str(row.get('Description', '')))
            feature_texts.append(f"{target} {desc}".strip(
            ) if target.lower() != 'nan' and target != '' else desc)

        embeddings = self.semantic_model.encode(feature_texts)
        amounts = df['Amount'].values.reshape(-1, 1)
        X = np.hstack((embeddings, amounts))

        preds = self.model.predict(X)
        labels = self.label_enc.inverse_transform(preds)

        final_labels = []
        for label, amt, text in zip(labels, df['Amount'], feature_texts):
            txt_lower = text.lower()

            if amt > 0:
                if label not in ["Transfers & Investments", "Other"]:
                    final_labels.append("Income")
                else:
                    final_labels.append(label)

            elif amt < 0:
                if any(k in txt_lower for k in ["tesco", "morrisons", "lidl", "asda", "sainsbury", "pret", "greggs"]):
                    final_labels.append("Food & Dining")
                elif any(k in txt_lower for k in ["uber", "trainline", "bee network", "stagecoach"]):
                    final_labels.append("Transportation")
                elif "lebara" in txt_lower:
                    final_labels.append("Bills & Utilities")
                elif label == "Income":
                    final_labels.append("Shopping")
                else:
                    final_labels.append(label)
            else:
                final_labels.append(label)

        df["Category"] = final_labels
        return df
