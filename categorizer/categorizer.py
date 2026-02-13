from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class Categorizer:
    def __init__(self, model_file_path, tokenizer_file_path, label_enc):
        self.categorizer_model = AutoModelForSequenceClassification.from_pretrained(
            model_file_path)
        self.categorizer_tokenizer = AutoTokenizer.from_pretrained(tokenizer_file_path)
        self.device = torch.device("mps") if torch.backends.mps.is_available() else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.label_enc = label_enc

        self.categorizer_model.to(self.device)

    def predict_category(self, texts):
        inputs = self.categorizer_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.categorizer_model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

        predicted_labels = self.label_enc.inverse_transform(preds.cpu().numpy())
        return predicted_labels
    
    def categorize_data(self, preprocessed_df):
        description = [str(desc) for desc in preprocessed_df["Description"]]
        target = [str(tg) for tg in preprocessed_df["Target Name"]]
        amount = [str(amt) for amt in preprocessed_df["Amount"]]

        text = []

        for desc, tg, amt in zip(description, target, amount):
            text.append(desc + " " + tg + " " + amt)

        classes = []

        for t in text:
            classes.append(self.predict_category(t))

        preprocessed_df["Category"] = classes

        return preprocessed_df
