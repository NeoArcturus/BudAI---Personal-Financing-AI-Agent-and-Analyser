import webbrowser
import os
from api_integrator.access_token_generator import AccessTokenGenerator
from api_integrator.get_account_detail import UserAccount
from categorizer.categorizer import Categorizer
from categorizer.preprocessor import Preprocessor
from training.model_trainer import CategorizerTrainer

access_token_gen = AccessTokenGenerator()
refresh_token = os.getenv("TRUELAYER_REFRESH_TOKEN")

refreshed = False
if refresh_token:
    refreshed = access_token_gen.regenerate_auth_token_using_refresh_token()

if not refreshed:
    try:
        auth_link = access_token_gen.get_auth_link()
        webbrowser.open(auth_link)
        access_token_gen.app.run(port=8080)
    except Exception as e:
        print(f"Error: {e}")

user_account = UserAccount()
user_account.get_user_account_details()
data = user_account.all_transactions(from_date="2024-01-01", to_date="2024-12-31")

preprocessor = Preprocessor(data)
clean_df, label_enc, _ = preprocessor.preprocess()

if not os.path.exists("saved_model/xgb_model.json"):
    trainer = CategorizerTrainer(clean_df, label_enc)
    trainer.train()

classifier = Categorizer(model_file_path="saved_model/xgb_model.json", label_enc_path="saved_label_enc/label_encoder.joblib")
classified_data = classifier.categorize_data(clean_df)
print(classified_data)