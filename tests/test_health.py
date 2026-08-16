from fastapi.testclient import TestClient

from app import main


def test_health_reports_models_present(monkeypatch):
    monkeypatch.setattr(main.aligner, "models_present", lambda: True)
    monkeypatch.setattr(main.aligner, "is_loaded", lambda: False)

    with TestClient(main.app) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["mfa"] == {"modelsPresent": True, "loaded": False}


def test_health_reports_models_missing(monkeypatch):
    monkeypatch.setattr(main.aligner, "models_present", lambda: False)
    monkeypatch.setattr(main.aligner, "is_loaded", lambda: False)

    with TestClient(main.app) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["mfa"] == {"modelsPresent": False, "loaded": False}
