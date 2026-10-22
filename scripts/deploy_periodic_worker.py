import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prefect.client.schemas.schedules import CronSchedule
from services.orchestration.periodic_worker import run_periodic_evaluation

if __name__ == "__main__":
    run_periodic_evaluation.serve(
        name="Periodic-Evaluator",
        schedule=CronSchedule(cron="*/5 * * * *"),
        tags=["ai-evaluation", "system-core"],
    )
    print("Successfully deployed Periodic-Evaluator to Prefect server.")
