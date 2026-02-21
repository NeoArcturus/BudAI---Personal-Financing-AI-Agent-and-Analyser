from sentence_transformers import SentenceTransformer
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
from tqdm import tqdm


class CategorizerTrainer:
    def __init__(self, clean_df, st_model_path):
        self.df = clean_df
        self.model = xgb.XGBClassifier(
            n_estimators=300, objective='multi:softprob')
        self.sentence_model = SentenceTransformer(st_model_path)

    def train(self):
        if 'Category' not in self.df.columns or self.df.empty:
            return
        with tqdm(total=4, desc="Training Brain") as pbar:
            le = LabelEncoder()
            y = le.fit_transform(self.df['Category'])
            pbar.update(1)
            embeddings = self.sentence_model.encode(
                self.df['Description'].tolist())
            X = np.hstack(
                (embeddings, self.df['Amount'].values.reshape(-1, 1)))
            pbar.update(1)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, stratify=y if len(np.unique(y)) > 1 else None)
            pbar.update(1)
            self.model.fit(X_train, y_train, eval_set=[
                           (X_val, y_val)], verbose=False)
            self.model.save_model("saved_model/xgb_model.json")
            joblib.dump(le, "saved_label_enc/label_encoder.joblib")
            pbar.update(1)
