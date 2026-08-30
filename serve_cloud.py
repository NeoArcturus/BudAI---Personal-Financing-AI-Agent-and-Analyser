from prefect import serve
from services.orchestration.onboarding_flows import process_user_onboarding_flow
from services.orchestration.cron_flows import (
    refresh_all_tokens_flow, 
    cloud_failsafe_sync_flow
)

if __name__ == "__main__":
    onboarding = process_user_onboarding_flow.to_deployment(
        name="cloud-onboarding"
    )
    token_refresh = refresh_all_tokens_flow.to_deployment(
        name="cloud-token-refresh", 
        cron="0 * * * *"
    )
    failsafe_sync = cloud_failsafe_sync_flow.to_deployment(
        name="cloud-failsafe-sync", 
        cron="0 0 * * *"
    )
    
    print("Serving Cloud Deployments...")
    serve(onboarding, token_refresh, failsafe_sync)
