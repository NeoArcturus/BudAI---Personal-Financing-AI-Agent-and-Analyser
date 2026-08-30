#!/usr/bin/env bash
set -e

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "[*] Initializing BudAI Automated Test Harness & Telemetry..."

# Spin up the lightweight OpenTelemetry dashboard (Jaeger)
echo "[*] Starting OpenTelemetry Collector..."
docker compose -f docker-compose.telemetry.yml up -d

# Export the collector endpoint so FastAPI and Pytest instrumentations route correctly
export COLLECTOR_ENDPOINT="http://localhost:4327"

# Clear previous test telemetry
rm -rf allure-results/

echo "[*] Phase 1: Executing Logical Suites (Unit & Integration)"
# Temporarily disable exit-on-error because TDD tests are expected to fail initially
set +e
pytest tests/unit tests/integration --alluredir=allure-results
PYTEST_EXIT_CODE=$?
set -e

echo "[*] Phase 2: Executing Headless Stress Test (Locust)"
echo "Simulating 1,000 concurrent TrueLayer webhooks over 30 seconds..."
# Run Locust in headless mode for automated execution
locust -f tests/stress/locustfile.py --host=http://localhost:8080 --headless -u 1000 -r 100 -t 30s

echo "[*] Phase 3: Telemetry & Tracing Dashboard"
echo "======================================================="
echo "The OpenTelemetry Tracing Dashboard is now live at:"
echo "http://localhost:16686"
echo "View this dashboard to inspect exact bottleneck stack traces from the load test."
echo "======================================================="

if [ $PYTEST_EXIT_CODE -ne 0 ]; then
    echo "[!] Pytest suite failed (Expected in TDD Red Phase). Launching Allure Dashboard to inspect logical failures..."
else
    echo "[*] All tests passed. Launching Allure Dashboard..."
fi

# Serve the interactive dashboard (This process binds to the terminal)
allure serve allure-results
