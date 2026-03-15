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
        state_uuid = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO auth_states (state_uuid, user_uuid) VALUES (?, ?)", (state_uuid, user_uuid))
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "scope": "info accounts balance cards transactions direct_debits standing_orders offline_access",
            "redirect_uri": self.redirect_uri,
            "providers": "uk-ob-all uk-oauth-all",
            "state": state_uuid
        }
        return f"{self.auth_base_url}?{urlencode(params)}"

    def generate_token_from_code(self, code, user_uuid):
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
            bank_uuid = str(uuid.uuid4())

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO banks (bank_uuid, truelayer_provider_id, user_uuid, bank_name, bank_logo_uri, access_token, refresh_token, consent_status, consent_status_updated_at, consent_created_at, consent_expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (bank_uuid, provider_id, user_uuid, provider_name, provider_logo_uri, enc_access, enc_refresh, consent_status, consent_status_updated_at, consent_created_at, consent_expires_at))
            return True

        return False

    def validate_callback(self, code, state):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT user_uuid FROM auth_states WHERE state_uuid = ?", (state,))
            row = cursor.fetchone()

        if not row:
            return False

        user_uuid = row[0]
        if self.generate_token_from_code(code, user_uuid):
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM auth_states WHERE state_uuid = ?", (state,))
            return True
        return False

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
                conn.execute("""
                    UPDATE banks SET access_token = ?, refresh_token = ? WHERE truelayer_provider_id = ? AND user_uuid = ?
                """, (enc_access, enc_refresh, provider_id, user_uuid))
            return res["access_token"]
        return None

    def extend_connection(self, refresh_token, user_has_reconfirmed_consent=True):
        payload = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "user_has_reconfirmed_consent": user_has_reconfirmed_consent,
            "refresh_token": refresh_token,
            "redirect_uri": self.redirect_uri,
            "user": {
                "id": str(uuid.uuid4()),
                "name": "Arnav Mishra",
                "email": "arnavpragya04@gmail.com"
            }
        }
        extend_url = "https://api.truelayer.com/data/v1/connections/extend"
        response = requests.post(extend_url, json=payload)
        return response.json()

    def extend_providers(self, provider_ids, user_uuid):
        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for p_id in provider_ids:
                cursor.execute(
                    "SELECT refresh_token FROM banks WHERE truelayer_provider_id = ? AND user_uuid = ?", (p_id, user_uuid))
                row = cursor.fetchone()
                if not row:
                    continue

                raw_refresh = self.cipher_suite.decrypt(row[0]).decode()
                res = self.extend_connection(raw_refresh, True)

                if res.get("action_needed") == "no_action_needed":
                    enc_access = self.cipher_suite.encrypt(
                        res["access_token"].encode())
                    enc_refresh = self.cipher_suite.encrypt(
                        res["refresh_token"].encode())
                    conn.execute(
                        "UPDATE banks SET access_token=?, refresh_token=? WHERE truelayer_provider_id=?", (enc_access, enc_refresh, p_id))
                    results.append({"provider_id": p_id, "status": "success"})
                elif res.get("action_needed") == "authentication_needed":
                    results.append({"provider_id": p_id, "status": "requires_reauth",
                                   "redirect_url": res.get("user_input_link")})
        return results

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

            enc_access = row[0]
            bank_uuid = row[1]

            try:
                raw_access = self.cipher_suite.decrypt(enc_access).decode()
                headers = {"Authorization": f"Bearer {raw_access}"}

                response = requests.delete(
                    "https://auth.truelayer.com/api/delete", headers=headers)

                if response.status_code in [204, 401, 200]:
                    if provider_id:
                        conn.execute(
                            "DELETE FROM transactions WHERE bank_uuid = ?", (bank_uuid,))
                        conn.execute(
                            "DELETE FROM accounts WHERE bank_uuid = ?", (bank_uuid,))
                        conn.execute(
                            "DELETE FROM banks WHERE bank_uuid = ?", (bank_uuid,))
                    else:
                        conn.execute(
                            "DELETE FROM transactions WHERE user_uuid = ?", (user_uuid,))
                        conn.execute(
                            "DELETE FROM accounts WHERE user_uuid = ?", (user_uuid,))
                        conn.execute(
                            "DELETE FROM banks WHERE user_uuid = ?", (user_uuid,))

                    results.append({"status": "revoked"})
                else:
                    results.append(
                        {"status": "failed", "truelayer_error": response.text})

            except Exception as e:
                results.append({"status": "error", "message": str(e)})

        return results
