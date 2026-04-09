import uuid
import os
import hashlib
import hmac
from config import SessionLocal
from models.database_models import User


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

    def _verify_password(self, password: str, stored_record: str) -> bool:
        if not stored_record:
            return False
        if "$" not in stored_record:
            # Backward compatibility for legacy plaintext records.
            return hmac.compare_digest(stored_record, password)
        salt_hex, stored_hash = stored_record.split("$", 1)
        calc_hash = self._hash_password(password, bytes.fromhex(salt_hex))
        return hmac.compare_digest(stored_hash, calc_hash)

    def authenticate_user(self, email, password):
        email = (email or "").strip().lower()
        if not email or not password:
            raise ValueError("Email and password are required.")

        with SessionLocal() as session:
            user = session.query(User).filter(User.name == email).first()
            if not user:
                raise ValueError("Invalid credentials.")
            if not self._verify_password(password, user.password or ""):
                raise ValueError("Invalid credentials.")
            return user.user_uuid

    def register_user(self, email, password):
        email = (email or "").strip().lower()
        if not email or not password:
            raise ValueError("Email and password are required.")

        with SessionLocal() as session:
            existing = session.query(User).filter(User.name == email).first()
            if existing:
                raise ValueError("Email already registered.")
            new_uuid = str(uuid.uuid4())
            new_user = User(
                user_uuid=new_uuid,
                name=email,
                password=self._build_password_record(password)
            )
            session.add(new_user)
            session.commit()
            return new_uuid
