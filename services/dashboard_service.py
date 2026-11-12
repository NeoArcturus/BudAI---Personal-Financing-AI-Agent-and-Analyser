import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy import text
from sqlmodel import Session, select

from config import SessionLocal
from models.database_models import (
    User, Bank, Account, Bucket, BucketType,
    SystemAlert, TransferSuggestion
)
from services.logger_setup import get_core_logger
from services.api_integrator.account_reader import AccountReader
from services.bucket_service import (
    get_or_create_unallocated_bucket,
    sync_unallocated_from_accounts,
    process_stage3_bills
)
from services.analytics.detection_pipelines import run_detection_pipelines
from services.analytics.surge_detector import get_surge_detector
from services.analytics.lifestyle_clustering import LifestyleClusteringService
from services.analytics.forecast_service import ForecasterEngine
from services.analytics.ons_client import get_ons_client

logger = get_core_logger(__name__)


class DashboardService:
    """
    Unified Dashboard Synthesizer Service.
    Assembles the complete state of user finances in a single execution pass:
    1. Connected bank accounts and physical cash anchor.
    2. Virtual buckets (UNALLOCATED, BILLS, GOALS).
    3. Stage 3 Intelligence (anomalies, burn rate, upcoming bills, forecaster curves).
    4. Deterministic lifestyle topology (Needs vs. Wants, 80/20 merchants).
    5. Active alerts and pending transfer suggestions.
    """

    def __init__(self):
        self.surge_detector = get_surge_detector()
        self.lifestyle_service = LifestyleClusteringService()
        self.forecaster_engine = ForecasterEngine()
        self.ons_client = get_ons_client()

    def get_full_snapshot(self, session: Session, user_uuid: str) -> Dict[str, Any]:
        logger.info(json.dumps({
            "message": f"Synthesizing complete dashboard snapshot for user {user_uuid}",
            "status_code": 200
        }))

        # -------------------------------------------------------------
        # 1. Connected Banks, Accounts, and Physical Cash Anchor
        # -------------------------------------------------------------
        account_reader = AccountReader(user_id=user_uuid)
        accounts_data = account_reader.get_all_accounts()

        total_physical_cash = 0.0
        currency_breakdown: Dict[str, float] = {}

        for acc in accounts_data:
            bal = float(acc.get("account_balance") or acc.get("balance") or 0.0)
            curr = str(acc.get("currency") or "GBP").upper()
            total_physical_cash += bal
            currency_breakdown[curr] = currency_breakdown.get(curr, 0.0) + bal

        # -------------------------------------------------------------
        # 2. Virtual Buckets (Strict 3 Types: UNALLOCATED, BILLS, GOALS)
        # -------------------------------------------------------------
        # Ensure unallocated bucket is synced to live bank accounts
        sync_unallocated_from_accounts(session, user_uuid)

        db_buckets = session.execute(
            select(Bucket).where(Bucket.user_id == user_uuid, Bucket.is_active == True)
        ).scalars().all()

        unallocated_data = None
        bills_data = []
        goals_data = []

        for b in db_buckets:
            b_dict = {
                "id": b.id,
                "name": b.name,
                "type": b.type.value if hasattr(b.type, "value") else str(b.type),
                "cached_balance": float(b.cached_balance or 0.0),
                "cached_balances": b.cached_balances or {},
                "currency": b.currency or "GBP",
                "priority": b.priority,
                "target_amount": float(b.target_amount) if b.target_amount else None,
                "target_date": b.target_date.isoformat() if b.target_date else None,
                "associated_account_id": b.associated_account_id,
                "associated_account_name": b.associated_account_name,
                "allocated_amount": float(b.allocated_amount or 0.0),
                "status_text": b.status_text,
                "has_shortfall_warning": bool(b.has_shortfall_warning),
                "shortfall_amount": float(b.shortfall_amount or 0.0)
            }
            b_type = b.type.value if hasattr(b.type, "value") else str(b.type)
            if b_type == BucketType.UNALLOCATED.value:
                unallocated_data = b_dict
            elif b_type == BucketType.BILLS.value:
                bills_data.append(b_dict)
            elif b_type == BucketType.GOALS.value:
                goals_data.append(b_dict)

        goals_data.sort(key=lambda x: (x["priority"] if x["priority"] is not None else 999))

        # -------------------------------------------------------------
        # 3. Stage 3 Intelligence (In-Memory Synthesizer)
        # -------------------------------------------------------------
        # 3a. Recurrence, subscriptions, and utilities
        discoveries = run_detection_pipelines(session, user_uuid)

        # 3b. Upcoming bills calendar
        calendar_arr, upcoming_bills = self.forecaster_engine.detect_upcoming_expenses(user_uuid)

        # Automatically process detected bills into Stage 4 buckets
        all_bills = (
            list(discoveries.get("new_subscriptions", [])) +
            list(discoveries.get("new_utilities", [])) +
            list(upcoming_bills or [])
        )
        try:
            process_stage3_bills(session, user_uuid, all_bills)
        except Exception as b_err:
            logger.warning(f"Auto-allocation pass skipped in snapshot: {b_err}")

        # 3c. Spend surge & Z-score status
        surge_status = self.surge_detector.check_daily_surge(user_uuid)

        # 3d. Deterministic lifestyle topology
        lifestyle_profile = self.lifestyle_service.analyze_user_lifestyle(user_uuid) or {}

        # 3e. Forecaster 30-day stochastic corridors (C++ Bates engine)
        forecast_corridors = {}
        try:
            primary_acc = accounts_data[0].get("account_id") if accounts_data else "PRIMARY"
            live_bal = float(accounts_data[0].get("account_balance") or 0.0) if accounts_data else 0.0
            S0, mu, sigma = self.forecaster_engine.fetch_and_calculate_parameters(primary_acc, live_bal, user_uuid)
            sim_df, _ = self.forecaster_engine.run_hybrid_simulation(
                account_id=primary_acc,
                S0=S0,
                mu=mu,
                user_uuid=user_uuid,
                days=30,
                paths=10000
            )
            if not sim_df.empty:
                forecast_corridors = {
                    "starting_balance": round(float(S0), 2),
                    "mean_trajectory": [round(float(v), 2) for v in sim_df["Mean"].tolist()],
                    "careless_bound": [round(float(v), 2) for v in sim_df["Careless"].tolist()],
                    "optimal_bound": [round(float(v), 2) for v in sim_df["Optimal"].tolist()]
                }
        except Exception as fe:
            logger.warning(f"Forecaster corridor fallback engaged: {fe}")
            forecast_corridors = {"starting_balance": total_physical_cash, "mean_trajectory": []}

        # 3f. Macroeconomic ONS benchmark
        macro_benchmark = {}
        try:
            ons_data = self.ons_client.sync_telemetry()
            macro_benchmark = {
                "national_cpih_rate": float(ons_data.get("latest_cpih", 3.4)),
                "updated_at": datetime.utcnow().strftime("%Y-%m-%d")
            }
        except Exception:
            macro_benchmark = {"national_cpih_rate": 3.4}

        # -------------------------------------------------------------
        # 4. Active Alerts and Transfer Suggestions
        # -------------------------------------------------------------
        db_alerts = session.execute(
            select(SystemAlert)
            .where(SystemAlert.user_id == user_uuid, SystemAlert.is_dismissed == False)
            .order_by(SystemAlert.timestamp.desc())
            .limit(10)
        ).scalars().all()

        alerts_data = [
            {
                "id": a.id,
                "title": a.title or a.event_type,
                "message": a.message,
                "urgency_level": a.urgency_level,
                "status_code": a.status_code,
                "is_read": bool(a.is_read),
                "timestamp": a.timestamp.isoformat() if a.timestamp else None
            }
            for a in db_alerts
        ]

        db_suggestions = session.execute(
            select(TransferSuggestion)
            .where(TransferSuggestion.user_uuid == user_uuid, TransferSuggestion.status != "COMPLETED")
            .order_by(TransferSuggestion.created_at.desc())
            .limit(5)
        ).scalars().all()

        suggestions_data = [
            {
                "id": s.suggestion_uuid,
                "from_account_id": s.from_account_id,
                "from_account_name": s.from_account_name,
                "to_account_id": s.to_account_id,
                "to_account_name": s.to_account_name,
                "amount": float(s.amount),
                "currency": s.currency,
                "reason": s.reason,
                "status": s.status,
                "verification_deadline": s.verification_deadline.isoformat() if s.verification_deadline else None,
                "created_at": s.created_at.isoformat() if s.created_at else None
            }
            for s in db_suggestions
        ]

        # -------------------------------------------------------------
        # 5. Recent Transactions Sample (Last 20)
        # -------------------------------------------------------------
        recent_txs = []
        try:
            reader = AccountReader(user_id=user_uuid)
            tx_df = reader.get_transactions(account_id=None, user_uuid=user_uuid)
            if tx_df is not None and not tx_df.empty:
                for _, r in tx_df.head(20).iterrows():
                    recent_txs.append({
                        "id": str(r.get("transaction_uuid", "")),
                        "description": str(r.get("semi_cleaned_description") or r.get("description", "")),
                        "amount": float(r.get("amount", 0.0)),
                        "category": str(r.get("category", "Uncategorized")),
                        "date": str(r.get("date", "")),
                        "account_id": str(r.get("account_id", ""))
                    })
        except Exception as te:
            logger.warning(f"Error fetching recent transactions sample: {te}")

        # -------------------------------------------------------------
        # Consolidated Full Snapshot Payload
        # -------------------------------------------------------------
        snapshot = {
            "user_uuid": user_uuid,
            "generated_at": datetime.utcnow().isoformat(),
            "financial_anchor": {
                "total_physical_cash": round(total_physical_cash, 2),
                "currency_breakdown": currency_breakdown,
                "connected_accounts": accounts_data
            },
            "buckets": {
                "unallocated": unallocated_data,
                "bills": bills_data,
                "goals": goals_data
            },
            "intelligence": {
                "upcoming_bills": upcoming_bills,
                "surge_status": surge_status,
                "lifestyle_profile": lifestyle_profile,
                "forecaster": forecast_corridors,
                "macro_benchmark": macro_benchmark
            },
            "alerts": alerts_data,
            "transfer_suggestions": suggestions_data,
            "recent_transactions": recent_txs
        }

        return snapshot


_dashboard_service_instance: Optional[DashboardService] = None

def get_dashboard_service() -> DashboardService:
    global _dashboard_service_instance
    if _dashboard_service_instance is None:
        _dashboard_service_instance = DashboardService()
    return _dashboard_service_instance
