import os
import xgboost as xgb
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


class CategorizerTrainer:
    def __init__(self, df, embeddings, model_dir, enc_dir):
        self.df = df
        self.embeddings = embeddings
        self.model = xgb.XGBClassifier(
            n_estimators=300, objective='multi:softprob', n_jobs=1)
        self.model_dir = model_dir
        self.enc_dir = enc_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.enc_dir, exist_ok=True)

    def train(self):
        if 'Category' not in self.df.columns or self.df.empty:
            return
        with tqdm(total=4, desc="Training Brain") as pbar:
            le = LabelEncoder()
            y = le.fit_transform(self.df['Category'])
            pbar.update(1)
            X = np.hstack(
                (self.embeddings, self.df['Amount'].values.reshape(-1, 1).astype(np.float32)))
            pbar.update(1)
            unique_labels, counts = np.unique(y, return_counts=True)
            if len(unique_labels) > 1 and np.all(counts >= 2):
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42)
            else:
                X_train, X_val, y_train, y_val = X, X, y, y
            pbar.update(1)
            self.model.fit(X_train, y_train, eval_set=[
                           (X_val, y_val)], verbose=False)
            self.model.save_model(os.path.join(
                self.model_dir, "xgb_model.json"))
            joblib.dump(le, os.path.join(self.enc_dir, "label_encoder.joblib"))
            pbar.update(1)
