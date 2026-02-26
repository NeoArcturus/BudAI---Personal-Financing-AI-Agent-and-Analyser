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
        self.account_id = os.getenv("TRUELAYER_ACCOUNT_ID", "")

    def get_user_account_details(self):
        """Fetches accounts and identifies the primary transaction account."""
        try:
            response = requests.get(self.base_url, headers=self.headers)
            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text}")
                return None

            all_accounts = response.json()
            results = all_accounts.get("results", [])

            for account in results:
                if account.get("account_type") == "TRANSACTION":
                    self.account_id = account.get("account_id")
                    dotenv.set_key(
                        ".env", "TRUELAYER_ACCOUNT_ID", self.account_id)
                    print(f"Found and saved Account ID: {self.account_id}")
                    return self.account_id

            if results:
                self.account_id = results[0].get("account_id")
                return self.account_id

        except Exception as e:
            print(f"Error fetching account details: {e}")
        return None

    def all_transactions(self, from_date, to_date):
        """Fetches transactions, automatically finding the account_id if missing."""

        if not self.account_id:
            print("Account ID not set. Searching for a valid transaction account...")
            if not self.get_user_account_details():
                print("Aborting: Could not find a valid account ID.")
                return None

        transactions_url = f"{self.base_url}/{self.account_id}/transactions"
        try:
            params = {"to": to_date, "from": from_date}
            response = requests.get(
                transactions_url, headers=self.headers, params=params)

            if response.status_code == 200:
                transactions = response.json()
                results = transactions.get("results", [])
                if results:
                    return pd.DataFrame(results)
                else:
                    print(
                        f"No transactions found for range {from_date} to {to_date}")
            else:
                print(
                    f"Failed to fetch transactions: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"Error fetching transactions: {e}")

    def get_account_balance(self):
        """Fetches the current balance of the account."""
        if not self.account_id:
            print("Account ID not set. Searching for a valid transaction account...")
            if not self.get_user_account_details():
                print("Aborting: Could not find a valid account ID.")
                return None

        balance_url = f"{self.base_url}/{self.account_id}/balance"
        try:
            response = requests.get(balance_url, headers=self.headers)
            if response.status_code == 200:
                balance_info = response.json()
                return balance_info.get("results", {})
            else:
                print(
                    f"Failed to fetch account balance: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"Error fetching account balance: {e}")
