import uuid
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
            # Fall back to local JWT verification
            pass

    # 2. Attempt verification via local HMAC secret
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

    # 3. Resolve user in Lakebase Postgres database
    user = db.query(User).filter(User.user_uuid == user_uuid).first()
    if not user and email:
        user = db.query(User).filter(User.email == email).first()

    # If user authenticated via Neon Auth but has no profile row yet, auto-provision
    if not user and email:
        user = User(
            user_uuid=user_uuid,
            email=email,
            password_hash="neon_managed_auth",
            first_name=email.split("@")[0].capitalize(),
            last_name=""
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return user
