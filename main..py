# from training.model_trainer import CategorizerTrainer
from categorizer.categorizer import Categorizer
from categorizer.preprocessor import Preprocessor

# trainer = CategorizerTrainer("statement_40878884_GBP_2024-01-01_2024-12-31.csv")
# trainer.train()

preprocessor = Preprocessor("statement_40878884_GBP_2024-01-01_2024-12-31.csv")

clean_df, label_enc, _ = preprocessor.preprocess()

classifier = Categorizer(model_file_path="saved_model", tokenizer_file_path="saved_tokenizer", label_enc=label_enc)

classified_data = classifier.categorize_data(clean_df)

print(classified_data)