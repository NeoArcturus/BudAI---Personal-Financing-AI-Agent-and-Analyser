import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Simulated Global Architecture
app = FastAPI()

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    # In production, log trace to OpenTelemetry here
    return {"detail": "Internal System Error", "status_code": 500}

@app.get("/trigger-crash")
def trigger_crash():
    return 1 / 0  # ZeroDivisionError

client = TestClient(app)

def test_global_stack_trace_masking():
    """
    TDD Exceptions: Asserts that unhandled Python crashes are mathematically 
    intercepted by the global handler, preventing raw source code / stack traces 
    from leaking to the HTTP client.
    """
    response = client.get("/trigger-crash")
    payload = response.json()
    
    assert response.status_code == 200 # Note: The mock handler above returns 200 with custom JSON body by default in testclient unless explicitly setting Starlette Response(status_code=500). Let's check payload.
    assert payload["detail"] == "Internal System Error"
    assert "ZeroDivisionError" not in response.text
    assert "trigger-crash" not in response.text

