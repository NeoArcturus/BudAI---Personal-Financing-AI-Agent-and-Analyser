import requests
import uuid
import logging
from urllib.parse import urlencode
from cryptography.fernet import Fernet
from sqlalchemy import text
from config import (

    SessionLocal,
    TRUELAYER_CLIENT_ID,
    TRUELAYER_CLIENT_SECRET,
    TRUELAYER_REDIRECT_URI,
    TRUELAYER_AUTH_URL,
    ENCRYPTION_KEY
)

import datetime
import pandas as pd
from models.database_models import Bank
from services.logger_setup import get_core_logger
logger = get_core_logger(__name__)

class AccessTokenGenerator:
    def __init__(self):
        self.client_id = TRUELAYER_CLIENT_ID
        self.client_secret = TRUELAYER_CLIENT_SECRET
        self.redirect_uri = TRUELAYER_REDIRECT_URI
        self.auth_base_url = TRUELAYER_AUTH_URL
        self.token_url = "https://auth.truelayer.com/connect/token"
        self.cipher_suite = Fernet(ENCRYPTION_KEY)
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
                if data.get("success") == True:
                    logger.info(
                        f"[BACKEND LOG] Re-auth URI successfully generated: {data.get('result')}")
                    return data.get("result")
            else:
                logger.error(
                    f"[ERROR] TrueLayer reauthuri returned {response.status_code}: {response.text}")
        except Exception as e:
            logger.error("An error occurred in this block", exc_info=True)
            logger.error(f"[ERROR] TrueLayer reauthuri generation failed: {e}")
        return None
    async def generate_token_from_code(self, code, state):
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
            def parse_tl_date(date_str):
                if not date_str:
                    return None
                try:
                    return pd.to_datetime(date_str, format='ISO8601').to_pydatetime()
                except Exception:
                    logger.error("An error occurred in this block", exc_info=True)
                    return datetime.now()
            updated_at = parse_tl_date(
                me_res['results'][0].get('consent_status_updated_at'))
            created_at = parse_tl_date(
                me_res['results'][0].get('consent_created_at'))
            expires_at = parse_tl_date(
                me_res['results'][0].get('consent_expires_at'))
            logger.info(f"{updated_at} {created_at} {expires_at}")
            enc_access = self.cipher_suite.encrypt(
                res["access_token"].encode())
            enc_refresh = self.cipher_suite.encrypt(
                res["refresh_token"].encode())
            from config import SessionLocal
            bank_id_to_init = None
            with SessionLocal() as session:
                bank = session.query(Bank).filter_by(
                    truelayer_provider_id=provider_id, user_uuid=state).first()
                if bank:
                    bank.access_token = enc_access
                    bank.refresh_token = enc_refresh
                    bank.consent_status = consent_status
                    bank.consent_status_updated_at = updated_at
                    bank.consent_created_at = created_at
                    bank.consent_expires_at = expires_at
                    bank.bank_name = provider_name
                    bank.bank_logo_uri = provider_logo_uri
                    bank_id_to_init = bank.bank_uuid
                else:
                    bank_id_to_init = str(uuid.uuid4())
                    new_bank = Bank(
                        bank_uuid=bank_id_to_init,
                        truelayer_provider_id=provider_id,
                        user_uuid=state,
                        bank_name=provider_name,
                        bank_logo_uri=provider_logo_uri,
                        access_token=enc_access,
                        refresh_token=enc_refresh,
                        consent_status=consent_status,
                        consent_status_updated_at=updated_at,
                        consent_created_at=created_at,
                        consent_expires_at=expires_at
                    )
                    session.add(new_bank)
                session.commit()
            logger.info(
                f"[AUTH LOG] Bank {provider_name} successfully linked/updated.")
            try:
                from services.api_integrator.get_account_detail import UserAccounts
                from fastapi_cache import FastAPICache
                import asyncio
                logger.info(
                    f"Dispatching background account initialization for {provider_name}...")
                user_acc = UserAccounts(user_id=state)
                loop = asyncio.get_running_loop()
                loop.run_in_executor(
                    None, user_acc.initialise_accounts, bank_id_to_init, state)
            except Exception as init_err:
                logger.error("An error occurred in this block", exc_info=True)
                logger.error(
                    f"Failed to trigger immediate initialization: {init_err}")
            return True
        else:
            logger.error(
                f"[AUTH ERROR] TrueLayer token exchange failed: {res}")
        return False
    async def validate_callback(self, code, state):
        return await self.generate_token_from_code(code, state)
    def refresh_token(self, provider_id, user_uuid):
        with SessionLocal() as session:
            row = session.execute(
                text("SELECT refresh_token FROM banks WHERE truelayer_provider_id = :provider_id AND user_uuid = :user_uuid"),
                {"provider_id": provider_id, "user_uuid": user_uuid}
            ).fetchone()
            if not row:
                return None
            refresh_token = self.cipher_suite.decrypt(bytes(row[0])).decode()
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
            with SessionLocal() as session:
                session.execute(
                    text("UPDATE banks SET access_token = :access, refresh_token = :refresh WHERE truelayer_provider_id = :provider_id AND user_uuid = :user_uuid"),
                    {"access": enc_access, "refresh": enc_refresh,
                        "provider_id": provider_id, "user_uuid": user_uuid}
                )
                session.commit()
            return res["access_token"]
        return None
    def revoke_provider(self, provider_id, user_uuid):
        results = []
        with SessionLocal() as session:
            if provider_id:
                row = session.execute(
                    text(
                        "SELECT access_token, bank_uuid FROM banks WHERE user_uuid = :user_uuid AND truelayer_provider_id = :provider_id"),
                    {"user_uuid": user_uuid, "provider_id": provider_id}
                ).fetchone()
            else:
                row = session.execute(
                    text(
                        "SELECT access_token, bank_uuid FROM banks WHERE user_uuid = :user_uuid LIMIT 1"),
                    {"user_uuid": user_uuid}
                ).fetchone()
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
                        session.execute(text("DELETE FROM transactions WHERE bank_uuid = :bank_uuid"), {
                                        "bank_uuid": bank_uuid})
                        session.execute(text("DELETE FROM accounts WHERE bank_uuid = :bank_uuid"), {
                                        "bank_uuid": bank_uuid})
                        session.execute(text("DELETE FROM banks WHERE bank_uuid = :bank_uuid"), {
                                        "bank_uuid": bank_uuid})
                    else:
                        session.execute(text("DELETE FROM transactions WHERE user_uuid = :user_uuid"), {
                                        "user_uuid": user_uuid})
                        session.execute(text("DELETE FROM accounts WHERE user_uuid = :user_uuid"), {
                                        "user_uuid": user_uuid})
                        session.execute(text("DELETE FROM banks WHERE user_uuid = :user_uuid"), {
                                        "user_uuid": user_uuid})
                    session.commit()
                    results.append({"status": "revoked"})
                else:
                    results.append(
                        {"status": "failed", "truelayer_error": response.text})
            except Exception as e:
                logger.error("An error occurred in this block", exc_info=True)
                results.append({"status": "error", "message": str(e)})
        return results
