import os
import requests
import uuid
import sqlite3
from urllib.parse import urlencode
from dotenv import load_dotenv
from cryptography.fernet import Fernet


class AccessTokenGenerator:
    def __init__(self, db_path="budai_memory.db"):
        load_dotenv()
        self.db_path = db_path
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.redirect_uri = os.getenv("REDIRECT_URI")
        self.auth_base_url = os.getenv("AUTH_LINK_URL")
        self.token_url = "https://auth.truelayer.com/connect/token"
        enc_key = os.getenv(
            "ENCRYPTION_KEY", b'cw_8H_1M4bX_3nF8vO5n3Y7A8xQ3_1m8aT2vP5_v5r8=')
        self.cipher_suite = Fernet(enc_key)

    def get_auth_link(self, user_uuid):
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "scope": "info accounts balance cards transactions direct_debits standing_orders offline_access",
            "redirect_uri": self.redirect_uri,
            "providers": "uk-ob-all uk-oauth-all",
            "state": user_uuid
        }
        return f"{self.auth_base_url}?{urlencode(params)}"

    def get_reauth_link(self, refresh_token, user_uuid):
        url = "https://auth.truelayer.com/v1/reauthuri"
        payload = {
            "refresh_token": refresh_token,
            "response_type": "code",
            "redirect_uri": self.redirect_uri
        }

        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                if data["success"] == True:
                    print(
                        f"[BACKEND LOG] Re-auth URI successfully generated: {data["result"]}")
                    return data["result"]
            else:
                print(
                    f"[ERROR] TrueLayer reauthuri returned {response.status_code}: {response.text}")
        except Exception as e:
            print(f"[ERROR] TrueLayer reauthuri generation failed: {e}")

        return None

    def generate_token_from_code(self, code, state):
        payload = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
            "code": code
        }
        response = requests.post(self.token_url, data=payload)
        res = response.json()

        if "access_token" in res:
            headers = {"Authorization": f"Bearer {res['access_token']}"}
            me_res = requests.get(
                "https://api.truelayer.com/data/v1/me", headers=headers).json()

            provider_id = me_res['results'][0]['provider']['provider_id']
            provider_name = me_res['results'][0]['provider']['display_name']
            provider_logo_uri = me_res['results'][0]['provider']['logo_uri']
            consent_status = me_res['results'][0]['consent_status']
            consent_status_updated_at = me_res['results'][0]['consent_status_updated_at']
            consent_created_at = me_res['results'][0]['consent_created_at']
            consent_expires_at = me_res['results'][0]['consent_expires_at']

            enc_access = self.cipher_suite.encrypt(
                res["access_token"].encode())
            enc_refresh = self.cipher_suite.encrypt(
                res["refresh_token"].encode())

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT bank_uuid, user_uuid FROM banks 
                    WHERE truelayer_provider_id = ?
                """, (provider_id,))
                existing_row = cursor.fetchone()

                if existing_row:
                    bank_uuid = existing_row[0]
                    user_uuid = existing_row[1]
                else:
                    bank_uuid = str(uuid.uuid4())
                    user_uuid = state

                cursor.execute("""
                    INSERT OR REPLACE INTO banks (bank_uuid, truelayer_provider_id, user_uuid, bank_name, bank_logo_uri, access_token, refresh_token, consent_status, consent_status_updated_at, consent_created_at, consent_expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (bank_uuid, provider_id, user_uuid, provider_name, provider_logo_uri, enc_access, enc_refresh, consent_status, consent_status_updated_at, consent_created_at, consent_expires_at))
                conn.commit()

            print(
                f"[AUTH LOG] Bank {provider_name} successfully linked/updated.")
            return True
        else:
            print(f"[AUTH ERROR] TrueLayer token exchange failed: {res}")

        return False

    def validate_callback(self, code, state):
        return self.generate_token_from_code(code, state)

    def refresh_token(self, provider_id, user_uuid):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT refresh_token FROM banks WHERE truelayer_provider_id = ? AND user_uuid = ?", (provider_id, user_uuid))
            row = cursor.fetchone()
            if not row:
                return None
            refresh_token = self.cipher_suite.decrypt(row[0]).decode()

        payload = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": refresh_token
        }
        response = requests.post(self.token_url, data=payload)
        res = response.json()
        if "access_token" in res:
            enc_access = self.cipher_suite.encrypt(
                res["access_token"].encode())
            enc_refresh = self.cipher_suite.encrypt(
                res.get("refresh_token", refresh_token).encode())
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE banks SET access_token = ?, refresh_token = ? WHERE truelayer_provider_id = ? AND user_uuid = ?
                """, (enc_access, enc_refresh, provider_id, user_uuid))
                conn.commit()
            return res["access_token"]
        return None

    def revoke_provider(self, provider_id, user_uuid):
        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if provider_id:
                cursor.execute(
                    "SELECT access_token, bank_uuid FROM banks WHERE user_uuid = ? AND truelayer_provider_id = ?", (user_uuid, provider_id))
            else:
                cursor.execute(
                    "SELECT access_token, bank_uuid FROM banks WHERE user_uuid = ? LIMIT 1", (user_uuid,))

            row = cursor.fetchone()
            if not row:
                return [{"status": "not_found", "error": "No connected accounts found"}]

            enc_access, bank_uuid = row[0], row[1]

            try:
                raw_access = self.cipher_suite.decrypt(enc_access).decode()
                headers = {"Authorization": f"Bearer {raw_access}"}

                response = requests.delete(
                    "https://auth.truelayer.com/api/delete", headers=headers)

                if response.status_code in [204, 401, 200]:
                    if provider_id:
                        cursor.execute(
                            "DELETE FROM transactions WHERE bank_uuid = ?", (bank_uuid,))
                        cursor.execute(
                            "DELETE FROM accounts WHERE bank_uuid = ?", (bank_uuid,))
                        cursor.execute(
                            "DELETE FROM banks WHERE bank_uuid = ?", (bank_uuid,))
                    else:
                        cursor.execute(
                            "DELETE FROM transactions WHERE user_uuid = ?", (user_uuid,))
                        cursor.execute(
                            "DELETE FROM accounts WHERE user_uuid = ?", (user_uuid,))
                        cursor.execute(
                            "DELETE FROM banks WHERE user_uuid = ?", (user_uuid,))

                    conn.commit()
                    results.append({"status": "revoked"})
                else:
                    results.append(
                        {"status": "failed", "truelayer_error": response.text})

            except Exception as e:
                results.append({"status": "error", "message": str(e)})

        return results
