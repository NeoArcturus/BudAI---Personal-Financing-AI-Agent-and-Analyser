from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Integer, LargeBinary, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from config import Base
from services.logger_setup import get_core_logger
logger = get_core_logger(__name__)

class User(Base):
    __tablename__ = "users"
    user_uuid = Column(String, primary_key=True, index=True)
    name = Column(String)
    password = Column(String)
    user_type = Column(String)
    banks = relationship("Bank", back_populates="user")
    accounts = relationship("Account", back_populates="user")
    transactions = relationship("Transaction", back_populates="user")
    liabilities = relationship("Liability", back_populates="user")
class Bank(Base):
    __tablename__ = "banks"
    bank_uuid = Column(String, primary_key=True, index=True)
    user_uuid = Column(String, ForeignKey("users.user_uuid"))
    truelayer_provider_id = Column(String)
    bank_name = Column(String)
    bank_logo_uri = Column(String)
    access_token = Column(LargeBinary)
    refresh_token = Column(LargeBinary)
    consent_status = Column(String)
    consent_status_updated_at = Column(DateTime)
    consent_created_at = Column(DateTime)
    consent_expires_at = Column(DateTime)
    user = relationship("User", back_populates="banks")
    accounts = relationship("Account", back_populates="bank")
class Account(Base):
    __tablename__ = "accounts"
    account_id = Column(String, primary_key=True, index=True)
    user_uuid = Column(String, ForeignKey("users.user_uuid"))
    bank_uuid = Column(String, ForeignKey("banks.bank_uuid"))
    account_number = Column(String)
    sort_code = Column(String)
    account_balance = Column(Float)
    user = relationship("User", back_populates="accounts")
    bank = relationship("Bank", back_populates="accounts")
    transactions = relationship("Transaction", back_populates="account")
class Transaction(Base):
    __tablename__ = "transactions"
    transaction_uuid = Column(String, primary_key=True, index=True)
    user_uuid = Column(String, ForeignKey("users.user_uuid"))
    bank_uuid = Column(String, ForeignKey("banks.bank_uuid"))
    account_id = Column(String, ForeignKey("accounts.account_id"))
    date = Column(DateTime)
    amount = Column(Float)
    category = Column(String)
    description = Column(String)
    user = relationship("User", back_populates="transactions")
    account = relationship("Account", back_populates="transactions")
class ChatSession(Base):
    __tablename__ = "chat_sessions"
    session_id = Column(String, primary_key=True, index=True)
    user_uuid = Column(String, ForeignKey("users.user_uuid"), index=True)
    title = Column(String)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    context_data = Column(JSON, nullable=True)
    messages = relationship("ChatHistory", back_populates="session", cascade="all, delete-orphan")
class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_uuid = Column(String, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.session_id"), index=True, nullable=True)
    role = Column(String)
    content = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    session = relationship("ChatSession", back_populates="messages")
class ForecastParameters(Base):
    __tablename__ = "forecast_parameters"
    user_uuid = Column(String, ForeignKey("users.user_uuid"), primary_key=True, index=True)
    kappa = Column(Float, default=2.0)
    theta = Column(Float, default=0.04)
    xi = Column(Float, default=0.1)
    rho = Column(Float, default=-0.5)
    lambda_val = Column(Float, default=0.1)
    mu_j = Column(Float, default=-0.05)
    sigma_j = Column(Float, default=0.1)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
class BackgroundTask(Base):
    __tablename__ = "background_tasks"
    task_id = Column(String, primary_key=True, index=True)
    user_uuid = Column(String, ForeignKey("users.user_uuid"), index=True)
    type = Column(String)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Liability(Base):
    __tablename__ = "liabilities"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_uuid = Column(String, ForeignKey("users.user_uuid"), index=True)
    name = Column(String)
    balance = Column(Float, default=0.0)
    interest_rate = Column(Float, default=0.0)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = relationship("User", back_populates="liabilities")

