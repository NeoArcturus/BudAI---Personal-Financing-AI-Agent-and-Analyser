import requests
import dotenv
import os
import pandas as pd

class UserAccount:
    def __init__(self):
        dotenv.load_dotenv()
        self.access_token = os.getenv("TRUELAYER_ACCESS_TOKEN")
        self.base_url = "https://api.truelayer.com/data/v1/accounts"

        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "accept": "application/json"
        }
        self.account_id = ""

    def get_user_account_details(self):
        try:
            response = requests.get(self.base_url, headers=self.headers)
            all_accounts = response.json()

            for account in all_accounts.get("results", []):
                if account.get("account_type") == "TRANSACTION":
                    self.account_id = account.get("account_id")
                    break
        except Exception as e:
            print(f"Error fetching account details: {e}")

    def all_transactions(self, from_date, to_date):
        if not self.account_id:
            print("No transaction account found.")
            return

        transactions_url = f"{self.base_url}/{self.account_id}/transactions/"
        try:
            params = {"to": to_date, "from": from_date}
            response = requests.get(transactions_url, headers=self.headers, params=params)
            transactions = response.json()
            if transactions.get("results"):
                df = pd.DataFrame(transactions["results"])
                df.to_csv("transactions.csv", index=False)
                print("Transactions saved to transactions.csv")
            else:
                print("No transactions found for the given date range.")
        except Exception as e:
            print(f"Error fetching transactions: {e}")
