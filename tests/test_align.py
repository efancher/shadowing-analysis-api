from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import aligner, audio, config, main


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


@contextmanager
def _fake_transcoded_wav(_audio_bytes: bytes):
    yield Path("/tmp/fake.wav")


def _mock_pipeline(monkeypatch, align_result=None, align_error=None):
    monkeypatch.setattr(main.audio, "transcoded_wav", _fake_transcoded_wav)

    def fake_align(_wav_path, _transcript):
        if align_error is not None:
            raise align_error
        return align_result

    monkeypatch.setattr(main.aligner, "align", fake_align)


def _files():
    return {"audio": ("clip.webm", b"fake-audio-bytes", "audio/webm")}


def test_align_rejects_blank_transcript(client):
    resp = client.post("/align", data={"transcript": "   "}, files=_files())
    assert resp.status_code == 422


def test_align_rejects_too_long_transcript(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_TRANSCRIPT_LENGTH", 5)
    resp = client.post("/align", data={"transcript": "今日はちょっと寒いですね"}, files=_files())
    assert resp.status_code == 422


def test_align_rejects_empty_audio(client):
    resp = client.post(
        "/align",
        data={"transcript": "今日はちょっと寒いですね"},
        files={"audio": ("clip.webm", b"", "audio/webm")},
    )
    assert resp.status_code == 422


def test_align_rejects_oversized_audio(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_AUDIO_BYTES", 4)
    resp = client.post("/align", data={"transcript": "今日は"}, files=_files())
    assert resp.status_code == 422


def test_align_returns_aligner_result(client, monkeypatch):
    result = {
        "durationSeconds": 1.7,
        "words": [{"start": 0.5, "end": 0.84, "text": "ちょっと", "phones": []}],
    }
    _mock_pipeline(monkeypatch, align_result=result)

    resp = client.post("/align", data={"transcript": "今日はちょっと寒いですね"}, files=_files())

    assert resp.status_code == 200
    assert resp.json() == result


def test_align_returns_503_when_aligner_unavailable(client, monkeypatch):
    _mock_pipeline(
        monkeypatch, align_error=aligner.AlignerUnavailableError("models missing")
    )
    resp = client.post("/align", data={"transcript": "今日は"}, files=_files())
    assert resp.status_code == 503


def test_align_returns_500_on_unexpected_alignment_failure(client, monkeypatch):
    _mock_pipeline(monkeypatch, align_error=RuntimeError("boom"))
    resp = client.post("/align", data={"transcript": "今日は"}, files=_files())
    assert resp.status_code == 500


def test_align_returns_422_when_audio_cannot_be_decoded(client, monkeypatch):
    @contextmanager
    def failing_transcode(_audio_bytes):
        raise audio.AudioTranscodeError("not a valid audio file")
        yield  # pragma: no cover - unreachable, keeps this a generator

    monkeypatch.setattr(main.audio, "transcoded_wav", failing_transcode)

    resp = client.post("/align", data={"transcript": "今日は"}, files=_files())
    assert resp.status_code == 422
