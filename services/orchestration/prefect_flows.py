import json
from prefect import flow, task
from config import SessionLocal
from services.orchestration.delta_sync import perform_physical_delta_sync
from services.logger_setup import get_core_logger, log_pipeline_data

logger = get_core_logger(__name__)

@task(retries=3, retry_delay_seconds=5)
def task_physical_delta_sync(user_uuid: str):
    with SessionLocal() as session:
        return perform_physical_delta_sync(session, user_uuid)

@flow(name="Agent_Evaluation_Loop")
def flow_agent_evaluation_loop(sync_result: dict, user_uuid: str):
    with SessionLocal() as session:
        # Phase 3: Gather and Mask
        from services.orchestration.data_masker import DataMasker
        from services.analytics.detection_pipelines import run_detection_pipelines
        from services.orchestration.context_builder import build_evaluation_context
        from services.orchestration.llm_executor import execute_llm_chain
        
        masker = DataMasker()
        masked_data = masker.mask_user_state(session, user_uuid)
        discoveries = run_detection_pipelines(session, user_uuid)
        
        log_pipeline_data(
            logger=logger,
            pipeline="agent_evaluation_loop",
            stage="context_prepared",
            data={
                "user_uuid": user_uuid,
                "sync_result": sync_result,
                "masked_buckets_count": len(masked_data.get("masked_state", [])),
                "discoveries": {
                    "subscriptions_count": len(discoveries.get("new_subscriptions", [])),
                    "utilities_count": len(discoveries.get("new_utilities", []))
                }
            },
            message=f"Agent Evaluation Loop context prepared with {len(masked_data.get('masked_state', []))} buckets"
        )
        
        brief = build_evaluation_context(
            session=session,
            user_uuid=user_uuid,
            masked_state=masked_data["masked_state"],
            discoveries=discoveries,
            trigger_event=sync_result
        )
        
        # Phase 4: Execute
        execute_llm_chain(
            user_uuid=user_uuid,
            compiled_brief=brief,
            reverse_map=masked_data["reverse_map"]
        )

@flow(name="Phase2_Event_Bus_Flow")
def flow_event_bus_sync(user_uuid: str):
    # 1. Delta Sync (The Algorithm)
    sync_result = task_physical_delta_sync(user_uuid)
    
    # 2. Handoff to Agentic Loop (Ringing the Alarm)
    if sync_result and sync_result.get("status") in ["synced", "aligned"]:
        flow_agent_evaluation_loop(sync_result, user_uuid)


