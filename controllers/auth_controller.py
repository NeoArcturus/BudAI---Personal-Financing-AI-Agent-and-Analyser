import json
from fastapi import HTTPException, Response, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone
from models.database_models import User, Bank
from models.status_codes import OpenBankingStatus
from config import (
    FRONTEND_URL, SessionLocal,
    TRUELAYER_CLIENT_ID, TRUELAYER_CLIENT_SECRET, TRUELAYER_REDIRECT_URI
)
from services.api_integrator.access_token_generator import AccessTokenGenerator
from services.logger_setup import get_core_logger
import httpx
import dateutil.parser

logger = get_core_logger(__name__)


async def get_user_profile(current_user: User):
    """Retrieves profile info for authenticated user."""
    return {
        "user_uuid": current_user.user_uuid,
        "name": getattr(current_user, 'name', 'User'),
        "email": getattr(current_user, 'email', ''),
        "date_of_birth": current_user.date_of_birth.isoformat() if current_user.date_of_birth else None,
        "employment_status": current_user.employment_status,
        "country_of_tax_residence": current_user.country_of_tax_residence,
        "persona": current_user.persona,
        "is_onboarded": current_user.is_onboarded
    }


async def get_truelayer_status(current_user: User, origin: str):
    """Generates TrueLayer authentication URL."""
    token_gen = AccessTokenGenerator()
    auth_url = token_gen.get_auth_link(current_user.user_uuid, origin)
    return {"auth_url": auth_url}


async def handle_truelayer_callback(code: str, state: str):
    """Handles OAuth callback from TrueLayer."""
    parts = state.split("::", 1)
    user_uuid = parts[0]
    target_frontend = parts[1] if len(parts) > 1 and parts[1] else FRONTEND_URL

    token_gen = AccessTokenGenerator()
    if await token_gen.validate_callback(code, user_uuid):
        from utils.cache_utils import clear_user_cache
        clear_user_cache(str(user_uuid), namespace="accounts")
        return RedirectResponse(f"{target_frontend}/home")

    logger.warning(json.dumps({"message": "TrueLayer callback validation failed", "status_code": 400}))
    raise HTTPException(status_code=400, detail="Authentication failed or session expired")


async def extend_connections(provider_ids: list, user_uuid: str):
    """Extends access for linked TrueLayer providers."""
    token_gen = AccessTokenGenerator()
    results = token_gen.extend_providers(provider_ids, user_uuid)
    return {"results": results}


async def revoke_access(provider_id: str, user_uuid: str):
    """Revokes access to a specific provider."""
    token_gen = AccessTokenGenerator()
    results = token_gen.revoke_provider(provider_id, user_uuid)
    return {"results": results}


async def generate_reauth_link_controller(bank_uuid: str, user_uuid: str):
    """Generates TrueLayer reauthentication link."""
    token_gen = AccessTokenGenerator()
    with SessionLocal() as session:
        bank = session.query(Bank).filter_by(bank_uuid=bank_uuid, user_uuid=user_uuid).first()
        if not bank:
            raise HTTPException(status_code=404, detail="Bank not found")

        enc_refresh = bank.refresh_token
        if isinstance(enc_refresh, memoryview):
            enc_refresh = enc_refresh.tobytes()

        refresh_token = token_gen.cipher_suite.decrypt(enc_refresh).decode()
        reauth_url = token_gen.get_reauth_link(refresh_token, user_uuid)

        if not reauth_url:
            raise HTTPException(status_code=500, detail="Failed to generate reauth link")

        return {"reauth_url": reauth_url}


async def get_truelayer_metadata(current_user: User):
    """Retrieves metadata and status of linked bank connections."""
    token_gen = AccessTokenGenerator()
    results = []

    with SessionLocal() as db:
        banks = db.query(Bank).filter(Bank.user_uuid == current_user.user_uuid).all()
        if not banks:
            return {"status": "success", "results": []}

        async with httpx.AsyncClient() as client:
            for bank in banks:
                if not bank.access_token or not bank.refresh_token:
                    continue
                try:
                    enc_token = bank.access_token
                    if isinstance(enc_token, memoryview):
                        enc_token = enc_token.tobytes()
                    access_token = token_gen.cipher_suite.decrypt(enc_token).decode()

                    enc_refresh = bank.refresh_token
                    if isinstance(enc_refresh, memoryview):
                        enc_refresh = enc_refresh.tobytes()
                    refresh_token = token_gen.cipher_suite.decrypt(enc_refresh).decode()

                    reauth_link = None
                    me_data = {}

                    extend_payload = {
                        "user_has_reconfirmed_consent": False,
                        "client_id": TRUELAYER_CLIENT_ID,
                        "client_secret": TRUELAYER_CLIENT_SECRET,
                        "refresh_token": refresh_token,
                        "redirect_uri": TRUELAYER_REDIRECT_URI,
                        "user": {
                            "id": current_user.user_uuid,
                            "name": getattr(current_user, "name", "User"),
                            "email": getattr(current_user, "email", "user@example.com")
                        }
                    }
                    ext_res = await client.post("https://api.truelayer.com/connections/extend", json=extend_payload)
                    action_needed = None

                    if ext_res.status_code == 200:
                        ext_data = ext_res.json()
                        action_needed = ext_data.get("action_needed")
                        if action_needed in ["authentication_needed", "reconfirmation_of_consent_needed"]:
                            bank.consent_status = OpenBankingStatus.CONNECTION_EXPIRED.value
                            reauth_link = ext_data.get("user_input_link")
                        elif action_needed == "no_action_needed":
                            bank.consent_status = OpenBankingStatus.CONNECTION_ACTIVE.value
                            if ext_data.get("access_token"):
                                bank.access_token = token_gen.cipher_suite.encrypt(ext_data["access_token"].encode())
                            if ext_data.get("refresh_token"):
                                bank.refresh_token = token_gen.cipher_suite.encrypt(ext_data["refresh_token"].encode())
                    else:
                        logger.warning(f"Extend connection failed for {bank.bank_name}: {ext_res.text}")

                    headers = {"Authorization": f"Bearer {access_token}"}
                    response = await client.get("https://api.truelayer.com/data/v1/me", headers=headers)

                    if response.status_code == 200:
                        data = response.json()
                        me_data = data.get("results", [{}])[0]
                        if me_data:
                            if me_data.get("consent_expires_at"):
                                bank.consent_expires_at = dateutil.parser.parse(me_data.get("consent_expires_at")).replace(tzinfo=None)
                            if me_data.get("consent_status_updated_at"):
                                bank.consent_status_updated_at = dateutil.parser.parse(me_data.get("consent_status_updated_at")).replace(tzinfo=None)

                            status_str = me_data.get("consent_status", "").lower()
                            if status_str == "revoked":
                                bank.consent_status = OpenBankingStatus.BANK_REVOKED_CONSENT.value
                            elif status_str == "expired":
                                bank.consent_status = OpenBankingStatus.CONNECTION_EXPIRED.value
                            elif status_str == "authorised":
                                if action_needed == "no_action_needed" and bank.consent_status not in [
                                    OpenBankingStatus.CONNECTION_EXPIRED.value,
                                    OpenBankingStatus.BANK_REVOKED_CONSENT.value
                                ]:
                                    bank.consent_status = OpenBankingStatus.CONNECTION_ACTIVE.value

                    results.append({
                        "bank_uuid": bank.bank_uuid,
                        "bank_name": bank.bank_name,
                        "consent_status": bank.consent_status,
                        "needs_reauth": bank.consent_status == OpenBankingStatus.CONNECTION_EXPIRED.value,
                        "reauth_link": reauth_link,
                        "metadata": me_data
                    })
                except Exception as e:
                    logger.error(f"Failed processing metadata for bank {bank.bank_uuid}: {e}")

        db.commit()
    return {"status": "success", "results": results}
