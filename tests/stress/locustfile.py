from locust import HttpUser, task, between
import uuid

class SandboxWebhookUser(HttpUser):
    """
    Simulates a high-concurrency burst of TrueLayer webhooks to test 
    Uvicorn worker exhaustion and Prefect event decoupling.
    """
    wait_time = between(0.1, 0.5)

    @task
    def trigger_refresh_webhook(self):
        # Generate a synthetic sandbox user ID
        synthetic_user_id = f"{uuid.uuid4()}@sandbox.budai.local"
        
        payload = {
            "event_id": str(uuid.uuid4()),
            "type": "Data.Refresh.Successful",
            "client_id": synthetic_user_id
        }
        
        # Test the FastAPI endpoint
        # The expected behavior is an instant 200/202, offloading the heavy work to Prefect.
        with self.client.post("/api/webhooks/truelayer", json=payload, catch_response=True) as response:
            if response.status_code not in [200, 202]:
                response.failure(f"Webhook failed with status {response.status_code}. Possible thread exhaustion.")
