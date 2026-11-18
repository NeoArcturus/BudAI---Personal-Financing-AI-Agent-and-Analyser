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
    """Retrieves metadata and status of linked bank connections safely without mutating consents."""
    token_gen = AccessTokenGenerator()
    results = []

    try:
        with SessionLocal() as db:
            banks = db.query(Bank).filter(Bank.user_uuid == current_user.user_uuid).all()
            if not banks:
                return {"status": "success", "results": []}

            async with httpx.AsyncClient(timeout=15.0) as client:
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

                        # 1. Query TrueLayer data/v1/me with current access token
                        headers = {"Authorization": f"Bearer {access_token}"}
                        response = await client.get("https://api.truelayer.com/data/v1/me", headers=headers)

                        # 2. If access token expired (401), attempt standard OAuth token refresh
                        if response.status_code == 401:
                            refresh_payload = {
                                "grant_type": "refresh_token",
                                "client_id": TRUELAYER_CLIENT_ID,
                                "client_secret": TRUELAYER_CLIENT_SECRET,
                                "refresh_token": refresh_token
                            }
                            ref_res = await client.post("https://auth.truelayer.com/connect/token", data=refresh_payload)
                            if ref_res.status_code == 200:
                                token_data = ref_res.json()
                                if "access_token" in token_data:
                                    access_token = token_data["access_token"]
                                    bank.access_token = token_gen.cipher_suite.encrypt(access_token.encode())
                                if "refresh_token" in token_data:
                                    refresh_token = token_data["refresh_token"]
                                    bank.refresh_token = token_gen.cipher_suite.encrypt(refresh_token.encode())

                                # Retry data/v1/me with fresh access token
                                headers = {"Authorization": f"Bearer {access_token}"}
                                response = await client.get("https://api.truelayer.com/data/v1/me", headers=headers)
                            else:
                                logger.warning(json.dumps({
                                    "message": f"Token refresh failed for {bank.bank_name}: {ref_res.status_code}",
                                    "status_code": ref_res.status_code
                                }))

                        # 3. Process metadata if data/v1/me returned 200
                        if response.status_code == 200:
                            data = response.json()
                            results_list = data.get("results", [])
                            if results_list:
                                me_data = results_list[0]

                                def _safe_parse_dt(dt_val):
                                    if not dt_val:
                                        return None
                                    try:
                                        return dateutil.parser.parse(dt_val).replace(tzinfo=None)
                                    except Exception:
                                        return None

                                if me_data.get("consent_expires_at"):
                                    bank.consent_expires_at = _safe_parse_dt(me_data.get("consent_expires_at"))
                                if me_data.get("consent_status_updated_at"):
                                    bank.consent_status_updated_at = _safe_parse_dt(me_data.get("consent_status_updated_at"))

                                status_str = me_data.get("consent_status", "").lower()
                                if status_str == "revoked":
                                    bank.consent_status = OpenBankingStatus.BANK_REVOKED_CONSENT.value
                                    reauth_link = token_gen.get_reauth_link(refresh_token, current_user.user_uuid)
                                elif status_str == "expired":
                                    bank.consent_status = OpenBankingStatus.CONNECTION_EXPIRED.value
                                    reauth_link = token_gen.get_reauth_link(refresh_token, current_user.user_uuid)
                                elif status_str == "authorised":
                                    now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
                                    if bank.consent_expires_at and bank.consent_expires_at <= now_naive:
                                        bank.consent_status = OpenBankingStatus.CONNECTION_EXPIRED.value
                                        reauth_link = token_gen.get_reauth_link(refresh_token, current_user.user_uuid)
                                    else:
                                        bank.consent_status = OpenBankingStatus.CONNECTION_ACTIVE.value
                        else:
                            bank.consent_status = OpenBankingStatus.CONNECTION_EXPIRED.value
                            reauth_link = token_gen.get_reauth_link(refresh_token, current_user.user_uuid)

                        needs_reauth = bank.consent_status in [
                            OpenBankingStatus.CONNECTION_EXPIRED.value,
                            OpenBankingStatus.BANK_REVOKED_CONSENT.value
                        ]

                        results.append({
                            "bank_uuid": bank.bank_uuid,
                            "bank_name": bank.bank_name,
                            "consent_status": bank.consent_status,
                            "needs_reauth": needs_reauth,
                            "reauth_link": reauth_link,
                            "metadata": me_data
                        })
                    except Exception as e:
                        logger.error(json.dumps({
                            "message": f"Failed processing metadata for bank {getattr(bank, 'bank_uuid', 'unknown')}: {e}",
                            "status_code": 500
                        }))
                        results.append({
                            "bank_uuid": getattr(bank, "bank_uuid", ""),
                            "bank_name": getattr(bank, "bank_name", "Unknown"),
                            "consent_status": getattr(bank, "consent_status", OpenBankingStatus.CONNECTION_EXPIRED.value),
                            "needs_reauth": True,
                            "reauth_link": None,
                            "metadata": {}
                        })

                try:
                    db.commit()
                except Exception as commit_err:
                    db.rollback()
                    logger.error(json.dumps({
                        "message": f"Database commit failed during metadata sync: {commit_err}",
                        "status_code": 500
                    }))

        return {"status": "success", "results": results}
    except Exception as e:
        logger.error(json.dumps({
            "message": f"Global error in get_truelayer_metadata: {e}",
            "status_code": 500
        }), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve connection metadata")
