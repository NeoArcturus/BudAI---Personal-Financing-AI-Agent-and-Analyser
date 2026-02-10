from training.model_trainer import CategorizerTrainer

trainer = CategorizerTrainer("statement_40878884_GBP_2024-01-01_2024-12-31.csv")
trainer.train()