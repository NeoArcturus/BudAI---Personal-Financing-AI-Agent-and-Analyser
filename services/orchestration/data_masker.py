import json
from sqlmodel import Session, select
from models.database_models import Bucket
from services.logger_setup import get_core_logger, log_pipeline_data

logger = get_core_logger(__name__)

class DataMasker:
    def __init__(self):
        self.forward_map = {}
        self.reverse_map = {}

    def mask_user_state(self, session: Session, user_uuid: str) -> dict:
        """
        Step 1 of Phase 3: Ephemeral Masking
        Creates temporary aliases for database UUIDs to protect PII before sending to the LLM.
        """
        buckets = session.execute(select(Bucket).where(Bucket.user_id == user_uuid).order_by(Bucket.priority_index)).scalars().all()
        
        masked_state = []
        for idx, bucket in enumerate(buckets):
            alias = f"BUCKET_{idx:03d}"
            self.forward_map[bucket.id] = alias
            self.reverse_map[alias] = bucket.id
            
            masked_state.append({
                "alias": alias,
                "type": bucket.type.value if hasattr(bucket.type, 'value') else str(bucket.type),
                "name": bucket.name,
                "balance": bucket.cached_balance,
                "target_amount": bucket.target_amount
            })
        
        log_pipeline_data(
            logger=logger,
            pipeline="data_masker",
            stage="mask_user_state",
            data={
                "user_uuid": user_uuid,
                "buckets_count": len(masked_state),
                "masked_buckets": [
                    {"alias": b["alias"], "type": b["type"], "balance": b["balance"]}
                    for b in masked_state
                ]
            },
            message=f"Ephemeral masking completed for {len(masked_state)} buckets"
        )
        
        return {
            "masked_state": masked_state,
            "reverse_map": self.reverse_map
        }
