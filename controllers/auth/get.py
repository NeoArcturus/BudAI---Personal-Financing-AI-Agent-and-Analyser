import json
from fastapi import HTTPException, Response
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from models.database_models import User
from config import FRONTEND_URL
from services.api_integrator.access_token_generator import AccessTokenGenerator
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

async def get_user_profile(current_user: User):
    """
    Retrieves the basic profile information for the authenticated user.
    
    Args:
        current_user (User): The authenticated user making the request.
        
    Returns:
        dict: The user's UUID, username, and email.
    """
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
    """
    Generates a TrueLayer authentication link to connect a new bank account.
    
    Args:
        current_user (User): The authenticated user making the request.
        origin (str): The frontend origin URL for redirection after auth.
        
    Returns:
        dict: A payload containing the generated TrueLayer auth_url.
    """
    token_gen = AccessTokenGenerator()
    auth_url = token_gen.get_auth_link(current_user.user_uuid, origin)
    return {"auth_url": auth_url}

async def handle_truelayer_callback(code: str, state: str):
    """
    Handles the OAuth callback from TrueLayer after a user links a bank account.
    Validates the authorization code, exchanges it for tokens, clears cache,
    and redirects the user back to the frontend.
    
    Args:
        code (str): The authorization code returned by TrueLayer.
        state (str): The state parameter containing user_uuid and target_frontend.
        
    Returns:
        RedirectResponse: Redirects the user back to the application home.
        
    Raises:
        HTTPException (400): If validation fails or the session is expired.
    """
    parts = state.split("::", 1)
    user_uuid = parts[0]
    target_frontend = parts[1] if len(parts) > 1 and parts[1] else FRONTEND_URL

    token_gen = AccessTokenGenerator()
    if await token_gen.validate_callback(code, user_uuid):
        from utils.cache_utils import clear_user_cache
        clear_user_cache(str(user_uuid), namespace="accounts")
        return RedirectResponse(f"{target_frontend}/home")
    
    logger.warning(json.dumps({"message": f"TrueLayer callback validation failed", "status_code": 400}))
    raise HTTPException(status_code=400, detail="Authentication failed or session expired")

async def get_truelayer_metadata(current_user: User):
    """
    Retrieves TrueLayer connection metadata for the user from the /me endpoint for ALL their bank connections.
    If SCA is expired or access is denied, uses the /connections/extend endpoint internally to determine if re-auth is needed.
    Updates the database with the latest consent status and expiration dates.
    """
    from config import SessionLocal, TRUELAYER_CLIENT_ID, TRUELAYER_CLIENT_SECRET, TRUELAYER_REDIRECT_URI
    from models.database_models import Bank
    from models.status_codes import OpenBankingStatus
    import httpx
    import dateutil.parser
    
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
                    
                    # 1. Try GET /me
                    headers = {"Authorization": f"Bearer {access_token}"}
                    response = await client.get("https://api.truelayer.com/data/v1/me", headers=headers)
                    
                    needs_reauth = False
                    reauth_link = None
                    me_data = {}
                    
                    if response.status_code == 200:
                        data = response.json()
                        me_data = data.get("results", [{}])[0]
                    elif response.status_code in [401, 403]:
                        # SCA likely expired, fallback to /connections/extend
                        needs_reauth = True
                    
                    # 2. If needed, call /connections/extend to check for reauth
                    if needs_reauth:
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
                        if ext_res.status_code == 200:
                            ext_data = ext_res.json()
                            action_needed = ext_data.get("action_needed")
                            if action_needed in ["authentication_needed", "reconfirmation_of_consent_needed"]:
                                bank.consent_status = OpenBankingStatus.CONNECTION_EXPIRED.value
                                reauth_link = ext_data.get("user_input_link")
                            elif action_needed == "no_action_needed":
                                bank.consent_status = OpenBankingStatus.CONNECTION_ACTIVE.value
                                # Update tokens if new ones were provided
                                if ext_data.get("access_token"):
                                    bank.access_token = token_gen.cipher_suite.encrypt(ext_data["access_token"].encode())
                                if ext_data.get("refresh_token"):
                                    bank.refresh_token = token_gen.cipher_suite.encrypt(ext_data["refresh_token"].encode())
                        else:
                            logger.warning(f"Extend connection failed for {bank.bank_name}: {ext_res.text}")
                            bank.consent_status = OpenBankingStatus.BANK_REVOKED_CONSENT.value
                    else:
                        if me_data:
                            status_str = me_data.get("consent_status", "").lower()
                            if status_str == "authorised":
                                bank.consent_status = OpenBankingStatus.CONNECTION_ACTIVE.value
                            elif status_str == "revoked":
                                bank.consent_status = OpenBankingStatus.BANK_REVOKED_CONSENT.value
                            elif status_str == "expired":
                                bank.consent_status = OpenBankingStatus.CONNECTION_EXPIRED.value
                            if me_data.get("consent_expires_at"):
                                bank.consent_expires_at = dateutil.parser.parse(me_data.get("consent_expires_at")).replace(tzinfo=None)
                            if me_data.get("consent_status_updated_at"):
                                bank.consent_status_updated_at = dateutil.parser.parse(me_data.get("consent_status_updated_at")).replace(tzinfo=None)
                    
                    # Compile result payload
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
