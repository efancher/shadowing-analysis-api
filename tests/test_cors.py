from fastapi.testclient import TestClient

from app import main


def test_allows_deployed_frontend_origin():
    with TestClient(main.app) as client:
        resp = client.get("/health", headers={"Origin": "https://efancher.github.io"})

    assert resp.headers["access-control-allow-origin"] == "https://efancher.github.io"


def test_rejects_unlisted_origin():
    with TestClient(main.app) as client:
        resp = client.get("/health", headers={"Origin": "https://evil.example.com"})

    assert "access-control-allow-origin" not in resp.headers
