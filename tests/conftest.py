import pytest
import asyncio
import httpx
from httpx import AsyncClient
from sqlalchemy.orm import sessionmaker

from main import app
from config import engine, get_db
from middleware.auth_middleware import get_current_user
from models.database_models import User

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
def db_session():
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()

def override_get_current_user():
    return User(user_uuid="test_sandbox_user_uuid", email="test@sandbox.budai.local")

@pytest.fixture(scope="function")
async def async_client(db_session):
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user

    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client
