import uuid
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from config import get_db, SECRET_KEY, ALGORITHM, NEON_AUTH_JWKS_URL
from models.database_models import User
from jose import jwt as jose_jwt
from jose.exceptions import ExpiredSignatureError as JoseExpiredError, JWTError as JoseJWTError
import jwt as pyjwt
from jwt import PyJWKClient
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

security = HTTPBearer()
optional_security = HTTPBearer(auto_error=False)

async def get_optional_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(optional_security), db: Session = Depends(get_db)) -> Optional[User]:
    if not credentials:
        return None
    try:
        return await get_current_user(credentials, db)
    except Exception:
        return None

jwk_client = PyJWKClient(NEON_AUTH_JWKS_URL) if NEON_AUTH_JWKS_URL else None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    token = credentials.credentials
    user_uuid = None
    email = None

    # 1. Attempt verification via Neon Auth Managed Better Auth JWKS if configured
    if jwk_client:
        try:
            signing_key = jwk_client.get_signing_key_from_jwt(token)
            payload = pyjwt.decode(
                token,
                signing_key.key,
                algorithms=["EdDSA", "RS256", "ES256"],
                options={"verify_aud": False}
            )
            user_uuid = payload.get("sub")
            email = payload.get("email")
            if user_uuid:
                logger.debug(f"Successfully authenticated via Neon Auth for sub: {user_uuid}")
        except Exception:
            # Fall back to session table or local JWT verification
            pass

    # 2. Check neon_auth.session table (if token is an opaque session token from Better Auth)
    if not user_uuid:
        try:
            from sqlalchemy import text
            from datetime import datetime, timezone
            session_row = db.execute(
                text('SELECT "userId", "expiresAt" FROM neon_auth.session WHERE token = :tok'),
                {"tok": token}
            ).mappings().first()
            if session_row:
                expires_at = session_row["expiresAt"]
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
                if expires_at > datetime.now(timezone.utc):
                    user_uuid = str(session_row["userId"])
                    logger.debug(f"Successfully authenticated via Neon Auth session table for userId: {user_uuid}")
        except Exception as e:
            logger.debug(f"Session table lookup failed: {e}")

    # 3. Attempt verification via local HMAC secret
    if not user_uuid:
        try:
            payload = jose_jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_uuid = payload.get("sub")
            email = payload.get("email")
        except JoseExpiredError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except JoseJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

    if not user_uuid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    # 4. Resolve user in Lakebase Postgres database
    user = db.query(User).filter(User.user_uuid == user_uuid).first()
    if not user and email:
        user = db.query(User).filter(User.email == email).first()

    # If user authenticated via Neon Auth but has no profile row yet, auto-provision
    if not user and user_uuid:
        user_name = None
        try:
            from sqlalchemy import text
            auth_user_row = db.execute(
                text('SELECT name, email FROM neon_auth."user" WHERE id = :uid::uuid'),
                {"uid": user_uuid}
            ).mappings().first()
            if auth_user_row:
                email = auth_user_row.get("email") or email
                user_name = auth_user_row.get("name")
        except Exception:
            pass

        user = User(
            user_uuid=user_uuid,
            email=email or f"{user_uuid}@authed.user",
            name=user_name or (email.split("@")[0].capitalize() if email else "User"),
            password=None,
            is_onboarded=False
        )
        db.add(user)
        try:
            db.commit()
            db.refresh(user)
        except Exception:
            db.rollback()
            user = db.query(User).filter(User.user_uuid == user_uuid).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return user
