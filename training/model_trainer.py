from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from torch.optim import AdamW
from tqdm import tqdm
import torch
from categorizer.preprocessor import Preprocessor


class CategorizerTrainer:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.trainee_model = None
        self.trainee_tokenizer = AutoTokenizer.from_pretrained(
            "distilbert/distilbert-base-uncased")
        self.device = torch.device("mps") if torch.backends.mps.is_available() else (
            torch.device("cuda") if torch.cuda.is_available(
            ) else torch.device("cpu")
        )
        self.sentence_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2")
        self.loss_fn = None
        self.num_labels = None

    class TransactionDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=256):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_len,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }

    def annotate_data(self, df):
        text = []
        target_names = df['Target Name'] if 'Target Name' in df.columns else df['Description']
        references = df['Payment Reference'] if 'Payment Reference' in df.columns else [
            ""] * len(df)
        amounts = df['Amount']

        for target, ref, amt in zip(target_names, references, amounts):
            t_str = str(target) if str(target) != 'nan' else "Unknown"
            r_str = str(ref) if str(ref) != 'nan' else "No Ref"

            row_text = f"[TAR] {t_str} [REF] {r_str}"

            try:
                val = float(amt)
                if val > 0:
                    amt_text = f"[AMT] {np.abs(val)} [DIR] in"
                else:
                    amt_text = f"[AMT] {np.abs(val)} [DIR] out"
            except:
                amt_text = "[AMT] 0 [DIR] out"

            text.append(f"{row_text} {amt_text}")

        return text

    def training_loop(self, model, optimizer, train_loader):
        epochs = 10
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch in progress_bar:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
                loss = self.loss_fn(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                progress_bar.set_postfix(
                    {"Train Loss": f"{total_loss/len(progress_bar):.4f}"})

    def train(self):
        preprocessor = Preprocessor(self.data_file_path)
        clean_df, label_enc, label = preprocessor.preprocess()
        classes = np.unique(label)

        class_weights = compute_class_weight(
            class_weight='balanced', classes=classes, y=label)
        weights = torch.tensor(
            class_weights, dtype=torch.float).to(self.device)

        text = self.annotate_data(clean_df)
        training_df = pd.DataFrame({"text": text, "label": label})

        num_labels = len(label_enc.classes_)
        self.num_labels = num_labels

        self.trainee_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert/distilbert-base-uncased",
            num_labels=num_labels
        )
        self.trainee_model.to(self.device)

        self.loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

        train_text, val_text, train_label, val_label = train_test_split(
            training_df["text"], training_df["label"], test_size=0.2, random_state=42, stratify=training_df["label"]
        )

        train_dataset = self.TransactionDataset(
            train_text.tolist(), train_label.tolist(), self.trainee_tokenizer)
        val_dataset = self.TransactionDataset(
            val_text.tolist(), val_label.tolist(), self.trainee_tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)

        optimizer = AdamW(self.trainee_model.parameters(), lr=2e-5)

        self.training_loop(self.trainee_model, optimizer, train_loader)

        self.trainee_model.eval()
        preds, true_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.trainee_model(
                    input_ids=input_ids, attention_mask=attention_mask)
                preds.extend(outputs.logits.argmax(dim=1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        all_label_ids = np.arange(len(label_enc.classes_))

        print(classification_report(true_labels, preds, labels=all_label_ids,
              target_names=label_enc.classes_, zero_division=0))

        self.trainee_model.save_pretrained("./saved_model")
        self.trainee_tokenizer.save_pretrained("./saved_tokenizer")
