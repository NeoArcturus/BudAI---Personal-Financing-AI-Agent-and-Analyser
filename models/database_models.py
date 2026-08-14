from typing import Optional, List, Any
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import UniqueConstraint, Column, JSON
from datetime import datetime
from services.logger_setup import get_core_logger
from models.status_codes import OpenBankingStatus, PipelineStatus, TaskStatus

logger = get_core_logger(__name__)

class User(SQLModel, table=True):
    __tablename__ = "users"
    user_uuid: str = Field(primary_key=True, index=True)
    name: Optional[str] = None
    password: Optional[str] = None
    user_type: Optional[str] = None
    
    banks: List["Bank"] = Relationship(back_populates="user")
    accounts: List["Account"] = Relationship(back_populates="user")
    transactions: List["Transaction"] = Relationship(back_populates="user")
    liabilities: List["Liability"] = Relationship(back_populates="user")
    budgets: List["Budget"] = Relationship(back_populates="user")
    allocation_rules: List["AllocationRule"] = Relationship(back_populates="user")
    subscriptions: List["Subscription"] = Relationship(back_populates="user")

class Bank(SQLModel, table=True):
    __tablename__ = "banks"
    bank_uuid: str = Field(primary_key=True, index=True)
    user_uuid: Optional[str] = Field(default=None, foreign_key="users.user_uuid")
    truelayer_provider_id: Optional[str] = None
    bank_name: Optional[str] = None
    bank_logo_uri: Optional[str] = None
    access_token: Optional[bytes] = None
    refresh_token: Optional[bytes] = None
    consent_status: Optional[str] = None
    consent_status_updated_at: Optional[datetime] = None
    consent_created_at: Optional[datetime] = None
    consent_expires_at: Optional[datetime] = None
    
    user: Optional["User"] = Relationship(back_populates="banks")
    accounts: List["Account"] = Relationship(back_populates="bank")

class Account(SQLModel, table=True):
    __tablename__ = "accounts"
    account_id: str = Field(primary_key=True, index=True)
    user_uuid: Optional[str] = Field(default=None, foreign_key="users.user_uuid")
    bank_uuid: Optional[str] = Field(default=None, foreign_key="banks.bank_uuid")
    account_number: Optional[str] = None
    sort_code: Optional[str] = None
    account_balance: Optional[float] = None
    currency: str = Field(default="GBP")
    account_type: Optional[str] = None
    display_name: Optional[str] = None
    last_synced_at: Optional[datetime] = None
    
    user: Optional["User"] = Relationship(back_populates="accounts")
    bank: Optional["Bank"] = Relationship(back_populates="accounts")
    transactions: List["Transaction"] = Relationship(back_populates="account")

class Transaction(SQLModel, table=True):
    __tablename__ = "transactions"
    __table_args__ = (
        UniqueConstraint('account_id', 'provider_transaction_id', name='uq_transaction_account_provider'),
    )
    transaction_uuid: str = Field(primary_key=True, index=True)
    user_uuid: Optional[str] = Field(default=None, foreign_key="users.user_uuid")
    bank_uuid: Optional[str] = Field(default=None, foreign_key="banks.bank_uuid")
    account_id: Optional[str] = Field(default=None, foreign_key="accounts.account_id")
    provider_transaction_id: Optional[str] = Field(default=None, index=True)
    date: Optional[datetime] = None
    amount: Optional[float] = None
    currency: str = Field(default="GBP")
    category: Optional[str] = None
    sub_category: Optional[str] = None
    description: Optional[str] = None
    semi_cleaned_description: Optional[str] = None
    fully_cleaned_description: Optional[str] = None
    is_semantic_anomaly: Optional[bool] = Field(default=False)
    tags: Optional[List[str]] = Field(default=None, sa_column=Column(JSON))
    
    user: Optional["User"] = Relationship(back_populates="transactions")
    account: Optional["Account"] = Relationship(back_populates="transactions")

class ChatSession(SQLModel, table=True):
    __tablename__ = "chat_sessions"
    session_id: str = Field(primary_key=True, index=True)
    user_uuid: Optional[str] = Field(default=None, foreign_key="users.user_uuid", index=True)
    title: Optional[str] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    context_data: Optional[Any] = Field(default=None, sa_column=Column(JSON))
    
    messages: List["ChatHistory"] = Relationship(back_populates="session", sa_relationship_kwargs={"cascade": "all, delete-orphan"})

class ChatHistory(SQLModel, table=True):
    __tablename__ = "chat_history"
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    user_uuid: Optional[str] = Field(default=None, index=True)
    session_id: Optional[str] = Field(default=None, foreign_key="chat_sessions.session_id", index=True)
    role: Optional[str] = None
    content: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ttft_ms: Optional[int] = None
    compute_time_ms: Optional[int] = None
    tokens: Optional[int] = None
    reasoning_content: Optional[str] = None
    
    session: Optional["ChatSession"] = Relationship(back_populates="messages")

class ForecastParameters(SQLModel, table=True):
    __tablename__ = "forecast_parameters"
    user_uuid: str = Field(primary_key=True, index=True, foreign_key="users.user_uuid")
    kappa: float = Field(default=2.0)
    theta: float = Field(default=0.04)
    xi: float = Field(default=0.1)
    rho: float = Field(default=-0.5)
    lambda_val: float = Field(default=0.1)
    mu_j: float = Field(default=-0.05)
    sigma_j: float = Field(default=0.1)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class BackgroundTask(SQLModel, table=True):
    __tablename__ = "background_tasks"
    task_id: str = Field(primary_key=True, index=True)
    user_uuid: Optional[str] = Field(default=None, foreign_key="users.user_uuid", index=True)
    type: Optional[str] = None
    status: str = Field(default="600-102") # 600-102 denotes a pending/processing task
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class Liability(SQLModel, table=True):
    __tablename__ = "liabilities"
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    user_uuid: Optional[str] = Field(default=None, foreign_key="users.user_uuid", index=True)
    name: Optional[str] = None
    balance: float = Field(default=0.0)
    interest_rate: float = Field(default=0.0)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    user: Optional["User"] = Relationship(back_populates="liabilities")

class AdvisorSummary(SQLModel, table=True):
    __tablename__ = "advisor_summaries"
    summary_uuid: str = Field(primary_key=True, index=True)
    user_uuid: Optional[str] = Field(default=None, foreign_key="users.user_uuid", index=True)
    widget_id: Optional[str] = Field(default=None, index=True)
    data_hash: Optional[str] = Field(default=None, index=True)
    summary_text: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChartCache(SQLModel, table=True):
    __tablename__ = "chart_cache"
    cache_id: str = Field(primary_key=True, index=True)
    chart_data: Optional[str] = None

class Budget(SQLModel, table=True):
    __tablename__ = "budgets"
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    user_uuid: Optional[str] = Field(default=None, foreign_key="users.user_uuid", index=True)
    category: Optional[str] = None
    monthly_limit: Optional[float] = None
    rollover_enabled: bool = Field(default=False)
    
    user: Optional["User"] = Relationship(back_populates="budgets")

class AllocationRule(SQLModel, table=True):
    __tablename__ = "allocation_rules"
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    user_uuid: Optional[str] = Field(default=None, foreign_key="users.user_uuid", index=True)
    bucket_name: Optional[str] = None
    percentage: Optional[float] = None
    
    user: Optional["User"] = Relationship(back_populates="allocation_rules")

class UserLifestyleProfile(SQLModel, table=True):
    __tablename__ = "user_lifestyle_profiles"
    profile_uuid: str = Field(primary_key=True, index=True)
    user_uuid: Optional[str] = Field(default=None, foreign_key="users.user_uuid", index=True)
    macro_persona: Optional[str] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
class LifestyleCluster(SQLModel, table=True):
    __tablename__ = "lifestyle_clusters"
    cluster_uuid: str = Field(primary_key=True, index=True)
    user_uuid: Optional[str] = Field(default=None, foreign_key="users.user_uuid", index=True)
    hdbscan_cluster_id: int
    micro_archetype: Optional[str] = None
    behavioral_summary: Optional[str] = None
    total_spend: Optional[float] = None
    transaction_count: Optional[int] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class Subscription(SQLModel, table=True):
    __tablename__ = "subscriptions"
    subscription_uuid: str = Field(primary_key=True, index=True)
    user_uuid: Optional[str] = Field(default=None, foreign_key="users.user_uuid", index=True)
    bank_uuid: Optional[str] = Field(default=None, foreign_key="banks.bank_uuid", index=True)
    account_id: Optional[str] = Field(default=None, foreign_key="accounts.account_id", index=True)
    merchant_name: str
    expected_amount: float
    last_payment_date: Optional[datetime] = None
    last_payment_amount: Optional[float] = None
    predicted_frequency: str
    next_expected_date: datetime
    is_price_hike: bool = Field(default=False)
    status: str = Field(default=PipelineStatus.SUBSCRIPTION_DETECTED.value)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    user: Optional["User"] = Relationship(back_populates="subscriptions")

class ProactiveInsight(SQLModel, table=True):
    __tablename__ = "proactive_insights"
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    user_uuid: str = Field(foreign_key="users.user_uuid", index=True)
    insight_text: str
    insight_type: str = Field(default="opportunity") # "warning", "opportunity", "info"
    urgency_level: int = Field(default=1) # 1=low, 5=high
    created_at: datetime = Field(default_factory=datetime.utcnow)

class MerchantRule(SQLModel, table=True):
    __tablename__ = "merchant_rules"
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    user_uuid: str = Field(foreign_key="users.user_uuid", index=True)
    merchant_name: str = Field(index=True)
    category: Optional[str] = None
    sub_category: Optional[str] = None
    tags: Optional[List[str]] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)
