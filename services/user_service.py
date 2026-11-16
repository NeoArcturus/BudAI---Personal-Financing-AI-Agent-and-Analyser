import uuid
import os
import hashlib
import hmac
import unicodedata
from sqlalchemy import text
from config import SessionLocal
from models.database_models import User
from services.logger_setup import get_core_logger
logger = get_core_logger(__name__)

class UserService:
    def __init__(self, db_path=None):
        pass

    def _hash_password(self, password: str, salt: bytes) -> str:
        pwd_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            120000
        )
        return pwd_hash.hex()

    def _build_password_record(self, password: str) -> str:
        salt = os.urandom(16)
        return f"{salt.hex()}${self._hash_password(password, salt)}"

    def _verify_better_auth_password(self, password: str, stored_hash: str) -> bool:
        """
        Verifies password against Neon Auth / Better Auth scrypt hash.
        Format: {salt_hex}:{derived_key_hex}
        Algorithm: scrypt(NFKC(password), salt, N=16384, r=16, p=1, dklen=64)
        """
        if not stored_hash or ":" not in stored_hash:
            return False
        try:
            salt_hex, expected_key_hex = stored_hash.split(":", 1)
            salt = bytes.fromhex(salt_hex)
            expected_key = bytes.fromhex(expected_key_hex)
            normalized_pwd = unicodedata.normalize("NFKC", password).encode("utf-8")
            derived_key = hashlib.scrypt(
                normalized_pwd,
                salt=salt,
                n=16384,
                r=16,
                p=1,
                maxmem=0x7fffffff,
                dklen=64
            )
            return hmac.compare_digest(derived_key, expected_key)
        except Exception as e:
            logger.error(f"Error verifying Better Auth password: {e}")
            return False

    def _verify_password(self, password: str, stored_record: str) -> bool:
        if not stored_record:
            return False
        # Neon / Better Auth format (salt_hex:key_hex)
        if ":" in stored_record:
            return self._verify_better_auth_password(password, stored_record)
        # Legacy local format (salt_hex$hash_hex)
        if "$" in stored_record:
            salt_hex, stored_hash = stored_record.split("$", 1)
            calc_hash = self._hash_password(password, bytes.fromhex(salt_hex))
            return hmac.compare_digest(stored_hash, calc_hash)
        return hmac.compare_digest(stored_record, password)

    def authenticate_user(self, email: str, password: str) -> str:
        email = (email or "").strip().lower()
        if not email or not password:
            raise ValueError("Email and password are required.")
        with SessionLocal() as session:
            user = session.query(User).filter(User.email == email).first()
            if not user:
                raise ValueError("Invalid credentials.")

            # Check neon_auth.account table for Better Auth credentials
            stored_password = None
            try:
                row = session.execute(
                    text("""
                        SELECT a.password 
                        FROM neon_auth.account a 
                        WHERE a."userId"::text = :user_uuid 
                          AND a."providerId" = 'credential'
                        LIMIT 1
                    """),
                    {"user_uuid": user.user_uuid}
                ).fetchone()
                if row and row[0]:
                    stored_password = row[0]
            except Exception as e:
                logger.warning(f"Failed to query neon_auth.account for {user.user_uuid}: {e}")

            if not stored_password:
                raise ValueError("Invalid credentials.")

            if not self._verify_password(password, stored_password):
                raise ValueError("Invalid credentials.")

            return user.user_uuid

    def register_user(self, email, password, name, date_of_birth=None, employment_status=None, country_of_tax_residence="UK"):
        email = (email or "").strip().lower()
        if not email or not password:
            raise ValueError("Email and password are required.")
        with SessionLocal() as session:
            existing = session.query(User).filter(User.email == email).first()
            if existing:
                raise ValueError("Email already registered.")
            new_uuid = str(uuid.uuid4())
            
            from datetime import datetime, timezone
            dob = None
            if date_of_birth:
                try:
                    dob = datetime.strptime(date_of_birth, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
            
            new_user = User(
                user_uuid=new_uuid,
                email=email,
                name=name,
                date_of_birth=dob,
                employment_status=employment_status,
                country_of_tax_residence=country_of_tax_residence
            )
            session.add(new_user)
            session.commit()
            return new_uuid
