from sentence_transformers import SentenceTransformer
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from categorizer.preprocessor import Preprocessor


class CategorizerTrainer:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=6,
            objective='multi:softprob',
            eval_metric='mlogloss',
            verbosity=1
        )
        self.sentence_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2")

    def train(self):
        preprocessor = Preprocessor(self.data_file_path)
        clean_df, global_le, global_labels = preprocessor.preprocess()

        y_strings = global_le.inverse_transform(global_labels)
        train_le = LabelEncoder()
        y = train_le.fit_transform(y_strings)

        embeddings = self.sentence_model.encode(
            clean_df['Description'].tolist(), show_progress_bar=True)
        amounts = clean_df['Amount'].values.reshape(-1, 1)
        X = np.hstack((embeddings, amounts))

        counts = Counter(y)
        rare_classes = [cls for cls, count in counts.items() if count < 2]

        if rare_classes:
            print(
                f"Detected {len(rare_classes)} rare categories. Adjusting data for training...")
            X_extra = []
            y_extra = []
            for cls in rare_classes:
                idx = np.where(y == cls)[0][0]
                X_extra.append(X[idx])
                y_extra.append(y[idx])

            X = np.vstack((X, np.array(X_extra)))
            y = np.concatenate((y, np.array(y_extra)))

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=50
        )

        preds = self.model.predict(X_val)
        print(classification_report(
            y_val,
            preds,
            labels=np.arange(len(train_le.classes_)),
            target_names=train_le.classes_,
            zero_division=0
        ))

        self.model.save_model("../saved_model/xgb_model.json")
        joblib.dump(train_le, "../saved_label_enc/label_encoder.joblib")
