from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_returns_app_message() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Hugging Face Tutorial Demos is running"


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_video_002_sentiment_route_exists_in_openapi() -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/api/video-002/pipeline-function/sentiment" in response.json()["paths"]
