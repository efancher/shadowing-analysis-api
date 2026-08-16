from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import asr, audio, config, main


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


@contextmanager
def _fake_transcoded_wav(_audio_bytes: bytes):
    yield Path("/tmp/fake.wav")


def _mock_pipeline(monkeypatch, text=None, error=None):
    monkeypatch.setattr(main.audio, "transcoded_wav", _fake_transcoded_wav)

    def fake_transcribe(_wav_path, _prompt):
        if error is not None:
            raise error
        return text

    monkeypatch.setattr(main.asr, "transcribe", fake_transcribe)


def _files():
    return {"audio": ("clip.webm", b"fake-audio-bytes", "audio/webm")}


def test_transcribe_rejects_empty_audio(client):
    resp = client.post("/transcribe", files={"audio": ("clip.webm", b"", "audio/webm")})
    assert resp.status_code == 422


def test_transcribe_rejects_oversized_audio(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_AUDIO_BYTES", 4)
    resp = client.post("/transcribe", files=_files())
    assert resp.status_code == 422


def test_transcribe_rejects_too_long_prompt(client):
    resp = client.post(
        "/transcribe", data={"prompt": "a" * (config.MAX_TRANSCRIPT_LENGTH + 1)}, files=_files()
    )
    assert resp.status_code == 422


def test_transcribe_returns_recognized_text(client, monkeypatch):
    _mock_pipeline(monkeypatch, text="今日はちょっと寒いですね")

    resp = client.post("/transcribe", data={"prompt": "今日はちょっと寒いですね"}, files=_files())

    assert resp.status_code == 200
    assert resp.json() == {"text": "今日はちょっと寒いですね"}


def test_transcribe_works_without_a_prompt(client, monkeypatch):
    _mock_pipeline(monkeypatch, text="hello")

    resp = client.post("/transcribe", files=_files())

    assert resp.status_code == 200


def test_transcribe_returns_503_when_asr_unavailable(client, monkeypatch):
    _mock_pipeline(monkeypatch, error=asr.AsrUnavailableError("model missing"))
    resp = client.post("/transcribe", files=_files())
    assert resp.status_code == 503


def test_transcribe_returns_500_on_unexpected_failure(client, monkeypatch):
    _mock_pipeline(monkeypatch, error=RuntimeError("boom"))
    resp = client.post("/transcribe", files=_files())
    assert resp.status_code == 500


def test_transcribe_returns_422_when_audio_cannot_be_decoded(client, monkeypatch):
    @contextmanager
    def failing_transcode(_audio_bytes):
        raise audio.AudioTranscodeError("not a valid audio file")
        yield  # pragma: no cover - unreachable, keeps this a generator

    monkeypatch.setattr(main.audio, "transcoded_wav", failing_transcode)

    resp = client.post("/transcribe", files=_files())
    assert resp.status_code == 422
