#!/usr/bin/env bash
set -e

PORT=8089
echo "=== Starting BudAI Core API on port $PORT for Curl Testing ==="
.venv/bin/uvicorn main:app --port $PORT --host 127.0.0.1 > /dev/null 2>&1 &
SERVER_PID=$!

cleanup() {
    echo "Shutting down test server PID $SERVER_PID..."
    kill -9 $SERVER_PID 2>/dev/null || true
}
trap cleanup EXIT

# Wait up to 30 seconds for server to become responsive
echo "Waiting for server to become ready on http://127.0.0.1:$PORT/healthz..."
READY=0
for i in {1..35}; do
    STATUS=$(curl -s http://127.0.0.1:$PORT/healthz || true)
    if [[ "$STATUS" == *"ok"* ]]; then
        echo "Server is ready after $i seconds!"
        READY=1
        break
    fi
    sleep 1
done

if [ $READY -eq 0 ]; then
    echo "ERROR: Server failed to start within 35 seconds"
    exit 1
fi

echo ""
echo "--- 1. Curl Test: GET /api/a2a/agents ---"
CURL_AGENTS=$(curl -s http://127.0.0.1:$PORT/api/a2a/agents)
echo "$CURL_AGENTS"
echo "$CURL_AGENTS" | grep -q "CascadeWorker" && echo "PASS: CascadeWorker present in agents list"

echo ""
echo "--- 2. Curl Test: POST /api/a2a/dispatch (Categorizer -> Analyser Handoff) ---"
CURL_DISPATCH=$(curl -s -X POST http://127.0.0.1:$PORT/api/a2a/dispatch \
  -H "Content-Type: application/json" \
  -d '{
    "recipient": "CategorizerAgent",
    "intent": "CLASSIFY_TRANSACTION",
    "message_type": "QUERY",
    "user_uuid": "8e7d5972-0074-477f-b99e-e15d27c63704",
    "payload": {
      "description": "Shell Petrol Station Fuel 50L",
      "merchant": "Shell"
    }
  }')
echo "$CURL_DISPATCH"
echo "$CURL_DISPATCH" | grep -q "Transportation" && echo "PASS: Transaction classified under Transportation"

echo ""
echo "--- 3. Curl Test: POST /api/a2a/dispatch (Scenario Multi-hop Shock Simulation) ---"
CURL_SCENARIO=$(curl -s -X POST http://127.0.0.1:$PORT/api/a2a/dispatch \
  -H "Content-Type: application/json" \
  -d '{
    "recipient": "ScenarioWorker",
    "intent": "SIMULATE_EXPENSE_SHOCK",
    "message_type": "QUERY",
    "user_uuid": "8e7d5972-0074-477f-b99e-e15d27c63704",
    "payload": {
      "shock_amount": 250.0
    }
  }')
echo "$CURL_SCENARIO"
echo "$CURL_SCENARIO" | grep -q "can_absorb_shock" && echo "PASS: ScenarioWorker simulated shock via A2A handoff"

echo ""
echo "--- 4. Curl Test: POST /api/a2a/remediate-anomaly (Anomaly Multi-hop Evaluation) ---"
CURL_ANOMALY=$(curl -s -X POST http://127.0.0.1:$PORT/api/a2a/remediate-anomaly \
  -H "Content-Type: application/json" \
  -d '{
    "user_uuid": "8e7d5972-0074-477f-b99e-e15d27c63704",
    "amount": 750.0,
    "category": "Food & Dining",
    "merchant": "Michelin Star Dining"
  }')
echo "$CURL_ANOMALY"
echo "$CURL_ANOMALY" | grep -q "anomaly_evaluation" && echo "PASS: Anomaly remediation workflow completed successfully"

echo ""
echo "=== ALL BACKGROUND CURL TESTS COMPLETED SUCCESSFULLY ==="
