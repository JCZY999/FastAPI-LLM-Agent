from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
import pytest

from app.agent import AgentFailure
from app.main import create_app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("APP_API_TOKEN", "test-application-token")
    monkeypatch.setattr("app.main.run_agent", AsyncMock(return_value=("Test-only answer", ["lookup_lesson"])))
    with TestClient(create_app()) as client:
        yield client


AUTH = {"Authorization": "Bearer test-application-token"}


def test_health_and_openapi(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert "/chat" in client.get("/openapi.json").json()["paths"]


def test_authentication(client):
    assert client.post("/chat", json={"message": "hello"}).status_code == 401
    assert client.post("/chat", headers={"Authorization": "Bearer wrong"}, json={"message": "hello"}).status_code == 401


@pytest.mark.parametrize("payload", [{"message": " "}, {"message": "x" * 2001}, {"message": 42}, {"message": "hi", "extra": 1}])
def test_input_validation(client, payload):
    assert client.post("/chat", headers=AUTH, json=payload).status_code == 422


def test_response_and_rate_limit(client):
    for _ in range(10):
        r = client.post("/chat", headers=AUTH, json={"message": "Explain FastAPI"})
        assert r.status_code == 200
        assert r.json()["tools_used"] == ["lookup_lesson"]
    assert client.post("/chat", headers=AUTH, json={"message": "hello"}).status_code == 429


@pytest.mark.parametrize("error,status", [(AgentFailure("private provider detail"), 502), (TimeoutError(), 504)])
def test_safe_errors(client, monkeypatch, error, status):
    monkeypatch.setattr("app.main.run_agent", AsyncMock(side_effect=error))
    r = client.post("/chat", headers=AUTH, json={"message": "hello"})
    assert r.status_code == status
    assert "private provider detail" not in r.text


def test_missing_configuration(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("APP_API_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with TestClient(create_app()) as client:
        assert client.post("/chat", json={"message": "hello"}).status_code == 503
