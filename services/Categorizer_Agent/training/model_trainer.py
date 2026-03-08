import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm


class CategorizerTrainer:
    def __init__(self, df, embeddings, model_dir, enc_dir):
        self.df = df
        self.embeddings = embeddings
        self.model = HistGradientBoostingClassifier(
            max_iter=500, random_state=42, min_samples_leaf=1)
        self.model_dir, self.enc_dir = model_dir, enc_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.enc_dir, exist_ok=True)

    def train(self):
        if 'Category' not in self.df.columns or self.df.empty:
            return
        with tqdm(total=4, desc="Training Brain") as pbar:
            le = LabelEncoder()
            y = le.fit_transform(self.df['Category'])
            pbar.update(1)

            self.df['Amount'] = pd.to_numeric(
                self.df['Amount'], errors='coerce').fillna(0)
            X = np.hstack(
                (self.embeddings, self.df['Amount'].values.reshape(-1, 1).astype(np.float32)))
            pbar.update(1)

            indices = np.arange(len(y))
            if len(np.unique(y)) > 1:
                idx_train, idx_val = train_test_split(
                    indices, test_size=0.2, random_state=42)
            else:
                idx_train, idx_val = indices, indices
            pbar.update(1)

            sw = compute_sample_weight(class_weight='balanced', y=y[idx_train])
            self.model.fit(X[idx_train], y[idx_train], sample_weight=sw)

            y_pred = self.model.predict(X[idx_val])
            if "Income" in le.classes_:
                inc_idx = le.transform(["Income"])[0]
                val_amts = self.df['Amount'].iloc[idx_val].values
                y_pred = np.where(val_amts > 0.001, inc_idx, y_pred)

            acc = accuracy_score(y[idx_val], y_pred)
            report = classification_report(
                y[idx_val], y_pred, target_names=le.classes_, zero_division=0)
            with open(os.path.join(self.model_dir, "classification_report.txt"), "w") as f:
                f.write(f"Accuracy: {acc*100:.2f}%\n\n{report}")

            joblib.dump(self.model, os.path.join(
                self.model_dir, "gbm_model.joblib"))
            joblib.dump(le, os.path.join(self.enc_dir, "label_encoder.joblib"))
            pbar.update(1)
