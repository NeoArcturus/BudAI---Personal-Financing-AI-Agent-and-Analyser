import re
import uuid
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from sqlalchemy import text
from sqlmodel import Session
from config import SessionLocal
from models.database_models import Transaction, MerchantKnowledge, Account, Bank
from services.analytics.anchor_service import clean_merchant_string
from services.logger_setup import get_core_logger, log_pipeline_data

logger = get_core_logger(__name__)

CARD_TXN_RE = re.compile(
    r"Card transaction of\s+(?P<total>[\d\.,]+)\s+(?P<foreign_curr>[A-Za-z]{3})\s+issued by\s+(?P<merchant>.*?)(?:\s+\(fee:\s*(?P<fee_amt>[\d\.,]+)\s*(?P<fee_curr>[A-Za-z]{3})\))?$",
    re.IGNORECASE
)
WISE_FEE_RE = re.compile(r"Wise Charges for:\s*(?P<ref>.*)", re.IGNORECASE)
GENERIC_FEE_RE = re.compile(r"(?:Charges|Fee)\s+for:\s*(?P<ref>.*)", re.IGNORECASE)
CONVERT_RE = re.compile(
    r"Converted\s+(?P<from_amt>[\d\.,]+)\s+(?P<from_curr>[A-Za-z]{3})\s+to\s+(?P<to_amt>[\d\.,]+)\s+(?P<to_curr>[A-Za-z]{3})(?:\s+\(fee:\s*(?P<fee_amt>[\d\.,]+)\s*(?P<fee_curr>[A-Za-z]{3})\))?",
    re.IGNORECASE
)


class MultiCurrencyReconciliationService:
    """
    Detects and reconciles multi-currency split transactions, multi-currency fee legs,
    and cross-currency pocket conversions belonging to the same financial institution (bank_uuid).
    
    Operates without altering or merging underlying PostgreSQL transaction rows, ensuring
    raw bank ledger reconciliation invariants remain strictly zero-sum. Ties transactions
    together under a unified MerchantKnowledge anchor and synchronizes categorization.
    """

    def __init__(self):
        pass

    def check_user_has_multicurrency_accounts(self, session: Session, user_uuid: str) -> List[str]:
        """
        Sub-millisecond pre-flight check: returns list of bank_uuids for the user
        that host accounts across more than one distinct currency.
        """
        query = text("""
            SELECT bank_uuid
            FROM accounts
            WHERE user_uuid = :user_uuid AND bank_uuid IS NOT NULL
            GROUP BY bank_uuid
            HAVING COUNT(DISTINCT currency) > 1
        """)
        rows = session.execute(query, {"user_uuid": user_uuid}).fetchall()
        return [r[0] for r in rows if r[0]]

    def reconcile_user_transactions(self, user_uuid: str) -> Dict[str, Any]:
        """
        Executes multi-currency reconciliation across all eligible banks for the given user.
        """
        with SessionLocal() as session:
            multi_currency_banks = self.check_user_has_multicurrency_accounts(session, user_uuid)
            if not multi_currency_banks:
                logger.debug(json.dumps({
                    "message": f"No multi-currency banks found for user {user_uuid}. Skipping reconciliation.",
                    "status_code": 100
                }))
                return {
                    "status": "success",
                    "reconciled_groups": 0,
                    "reconciled_transactions": 0,
                    "message": "No transactions to reconcile."
                }

            total_reconciled_groups = 0
            total_reconciled_txs = 0
            group_details = []

            for bank_uuid in multi_currency_banks:
                res = self._reconcile_bank_transactions(session, user_uuid, bank_uuid)
                total_reconciled_groups += res["groups_count"]
                total_reconciled_txs += res["txs_count"]
                group_details.extend(res["details"])

            session.commit()

            message = (
                f"Reconciled {total_reconciled_txs} transactions across {total_reconciled_groups} multi-currency events."
                if total_reconciled_groups > 0
                else "No transactions to reconcile."
            )

            log_pipeline_data(
                logger=logger,
                pipeline="multicurrency_reconciliation",
                stage="completed",
                data={
                    "user_uuid": user_uuid,
                    "multi_currency_banks": multi_currency_banks,
                    "reconciled_groups": total_reconciled_groups,
                    "reconciled_transactions": total_reconciled_txs
                },
                message=message
            )

            return {
                "status": "success",
                "reconciled_groups": total_reconciled_groups,
                "reconciled_transactions": total_reconciled_txs,
                "message": message,
                "details": group_details
            }

    def _reconcile_bank_transactions(
        self,
        session: Session,
        user_uuid: str,
        bank_uuid: str
    ) -> Dict[str, Any]:
        """
        Scans transactions for a single multi-currency bank_uuid, clusters by timestamp (<= 2s),
        and applies manifest reconciliation.
        """
        query = text("""
            SELECT transaction_uuid, account_id, bank_uuid, date, amount, currency, description,
                   merchant_knowledge_uuid, sub_category, tags
            FROM transactions
            WHERE user_uuid = :user_uuid AND bank_uuid = :bank_uuid
            ORDER BY date ASC
        """)
        rows = session.execute(query, {"user_uuid": user_uuid, "bank_uuid": bank_uuid}).mappings().all()
        if not rows or len(rows) < 2:
            return {"groups_count": 0, "txs_count": 0, "details": []}

        tx_list = [dict(r) for r in rows]

        # 1. Cluster by timestamp window (<= 2.0 seconds)
        clusters: List[List[Dict[str, Any]]] = []
        current_cluster: List[Dict[str, Any]] = []

        for tx in tx_list:
            if not current_cluster:
                current_cluster.append(tx)
            else:
                delta = abs((tx["date"] - current_cluster[-1]["date"]).total_seconds())
                if delta <= 2.0:
                    current_cluster.append(tx)
                else:
                    if len(current_cluster) > 1:
                        clusters.append(current_cluster)
                    current_cluster = [tx]
        if len(current_cluster) > 1:
            clusters.append(current_cluster)

        reconciled_groups = 0
        reconciled_txs = 0
        details = []

        for cluster in clusters:
            match_res = self._match_and_reconcile_cluster(session, user_uuid, bank_uuid, cluster)
            if match_res:
                reconciled_groups += 1
                reconciled_txs += len(match_res["transaction_uuids"])
                details.append(match_res)

        return {
            "groups_count": reconciled_groups,
            "txs_count": reconciled_txs,
            "details": details
        }

    def _match_and_reconcile_cluster(
        self,
        session: Session,
        user_uuid: str,
        bank_uuid: str,
        cluster: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates a candidate timestamp cluster against multi-currency split & fee patterns.
        """
        # Distinguish legs inside cluster
        funding_leg = None
        drained_leg = None
        fee_leg = None
        parsed_funding_info = None

        for tx in cluster:
            desc = (tx.get("description") or "").strip()
            
            # Check for Fee pattern
            if WISE_FEE_RE.search(desc) or GENERIC_FEE_RE.search(desc):
                fee_leg = tx
                continue

            # Check for Card transaction pattern
            m = CARD_TXN_RE.search(desc)
            if m:
                info = m.groupdict()
                if info.get("fee_amt"):
                    funding_leg = tx
                    parsed_funding_info = info
                else:
                    drained_leg = tx
                continue

            # Check for direct Conversion pattern
            cm = CONVERT_RE.search(desc)
            if cm:
                cinfo = cm.groupdict()
                if cinfo.get("fee_amt") or tx.get("amount", 0) < 0:
                    funding_leg = tx
                    parsed_funding_info = {
                        "merchant": "Wise Currency Conversion",
                        "total": cinfo.get("from_amt"),
                        "foreign_curr": cinfo.get("from_curr"),
                        "fee_amt": cinfo.get("fee_amt"),
                        "fee_curr": cinfo.get("fee_curr")
                    }
                else:
                    drained_leg = tx

        # We must have at least a funding leg and a fee leg, or a funding leg and a drained leg
        if not funding_leg or (not fee_leg and not drained_leg):
            return None

        # Verify fee amount match if fee leg is present
        if fee_leg and parsed_funding_info and parsed_funding_info.get("fee_amt"):
            expected_fee = float(parsed_funding_info["fee_amt"].replace(",", ""))
            actual_fee = abs(float(fee_leg["amount"] or 0.0))
            if abs(expected_fee - actual_fee) > 0.01:
                # Fee mismatch, do not link
                return None

        # Resolve or create canonical MerchantKnowledge
        raw_merchant = parsed_funding_info.get("merchant") if parsed_funding_info else funding_leg.get("description")
        clean_name = clean_merchant_string(raw_merchant)

        merchant_knowledge = self._get_or_create_merchant_knowledge(
            session=session,
            clean_name=clean_name,
            funding_tx=funding_leg
        )

        all_cluster_txs = [t for t in [funding_leg, drained_leg, fee_leg] if t is not None]
        tx_uuids = [t["transaction_uuid"] for t in all_cluster_txs]

        # Update each transaction leg in the database
        for tx_dict in all_cluster_txs:
            tx_obj = session.query(Transaction).filter_by(
                transaction_uuid=tx_dict["transaction_uuid"]
            ).first()
            if not tx_obj:
                continue

            # 1. Set unified merchant_knowledge_uuid
            tx_obj.merchant_knowledge_uuid = merchant_knowledge.knowledge_uuid

            # 2. Inherit sub_category
            if merchant_knowledge.sub_category:
                tx_obj.sub_category = merchant_knowledge.sub_category

            # 3. Synchronize tags
            existing_tags = set(tx_obj.tags or [])
            
            if tx_dict == fee_leg:
                # Remove penalty/overdraft tags, add dedicated FX fee tags
                existing_tags.discard("#penalty")
                existing_tags.discard("#bank_charges")
                existing_tags.discard("#overdraft")
                existing_tags.add("#fx_fee")
                existing_tags.add("#card_fee")
            else:
                existing_tags.add("#split_payment")
                existing_tags.add("#multi_currency")
                if merchant_knowledge.tags:
                    for t in merchant_knowledge.tags:
                        existing_tags.add(t)

            tx_obj.tags = sorted(list(existing_tags))

        # Calculate implied conversion rate and total in base currency
        foreign_total = float(parsed_funding_info["total"].replace(",", "")) if parsed_funding_info and parsed_funding_info.get("total") else 0.0
        foreign_currency = parsed_funding_info.get("foreign_curr") if parsed_funding_info else funding_leg["currency"]

        base_spent = abs(float(funding_leg["amount"] or 0.0))
        fee_spent = abs(float(fee_leg["amount"] or 0.0)) if fee_leg else 0.0
        drained_spent = abs(float(drained_leg["amount"] or 0.0)) if drained_leg else 0.0

        return {
            "merchant_name": clean_name,
            "merchant_knowledge_uuid": merchant_knowledge.knowledge_uuid,
            "category": merchant_knowledge.category,
            "sub_category": merchant_knowledge.sub_category,
            "transaction_uuids": tx_uuids,
            "foreign_total": foreign_total,
            "foreign_currency": foreign_currency,
            "base_currency": funding_leg["currency"],
            "base_spent": base_spent,
            "fee_spent": fee_spent,
            "drained_spent": drained_spent,
            "date": funding_leg["date"].isoformat() if funding_leg.get("date") else None
        }

    def _get_or_create_merchant_knowledge(
        self,
        session: Session,
        clean_name: str,
        funding_tx: Dict[str, Any]
    ) -> MerchantKnowledge:
        """
        Retrieves existing MerchantKnowledge anchor or constructs a new one.
        """
        # 1. Check if an anchor already exists for this exact clean merchant name
        existing = session.query(MerchantKnowledge).filter(
            MerchantKnowledge.clean_merchant_name == clean_name
        ).first()

        if existing:
            return existing

        # 2. Check if funding_tx already points to a valid anchor
        if funding_tx.get("merchant_knowledge_uuid"):
            anchor = session.query(MerchantKnowledge).filter_by(
                knowledge_uuid=funding_tx["merchant_knowledge_uuid"]
            ).first()
            if anchor:
                if anchor.clean_merchant_name != clean_name and clean_name != "Unknown Merchant":
                    anchor.clean_merchant_name = clean_name
                return anchor

        # 3. Create a new MerchantKnowledge record
        k_uuid = str(uuid.uuid4())
        category = "Subscriptions & Digital Services" if "Recharge" in clean_name or "Google" in clean_name else "Shopping"
        sub_category = funding_tx.get("sub_category") or "General Discretionary"
        tags = ["#discretionary", "#online_service", "#multi_currency"]

        mk = MerchantKnowledge(
            knowledge_uuid=k_uuid,
            clean_merchant_name=clean_name,
            category=category,
            sub_category=sub_category,
            tags=tags,
            is_human_verified=False,
            created_at=datetime.now(timezone.utc)
        )
        session.add(mk)
        session.flush()

        # Index to Qdrant vector database in background if available
        try:
            from services.embedding_service import get_embedding_service
            from services.qdrant_service import get_qdrant_service
            embedder = get_embedding_service()
            qdrant = get_qdrant_service()
            vec = embedder.embed_text(clean_name)
            qdrant.upsert_merchant(
                knowledge_uuid=k_uuid,
                vector=vec,
                clean_name=clean_name,
                category=category,
                sub_category=sub_category,
                tags=tags,
                is_human_verified=False
            )
        except Exception as q_err:
            logger.debug(f"Qdrant indexing deferred for new merchant {clean_name}: {q_err}")

        return mk


_reconciliation_service: Optional[MultiCurrencyReconciliationService] = None

def get_multicurrency_reconciliation_service() -> MultiCurrencyReconciliationService:
    global _reconciliation_service
    if _reconciliation_service is None:
        _reconciliation_service = MultiCurrencyReconciliationService()
    return _reconciliation_service
