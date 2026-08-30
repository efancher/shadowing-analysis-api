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


def _mock_source_pipeline(monkeypatch, segments=None, error=None):
    monkeypatch.setattr(main.audio, "transcoded_wav", _fake_transcoded_wav)

    def fake_transcribe_source(_wav_path):
        if error is not None:
            raise error
        return segments

    monkeypatch.setattr(main.asr, "transcribe_source", fake_transcribe_source)


def test_transcribe_source_returns_timed_segments(client, monkeypatch):
    segs = [
        {"text": "先生が本を読んでいた。", "startMs": 0, "endMs": 2500,
         "avgLogprob": -0.3, "noSpeechProb": 0.01},
        {"text": "静かな部屋だった。", "startMs": 2500, "endMs": 4800,
         "avgLogprob": -0.4, "noSpeechProb": 0.02},
    ]
    _mock_source_pipeline(monkeypatch, segments=segs)
    resp = client.post("/transcribe-source", files=_files())
    assert resp.status_code == 200
    assert resp.json() == {"segments": segs}


def test_transcribe_source_rejects_empty_audio(client):
    resp = client.post(
        "/transcribe-source", files={"audio": ("s.opus", b"", "audio/ogg")}
    )
    assert resp.status_code == 422


def test_transcribe_source_rejects_oversized_audio(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_SOURCE_AUDIO_BYTES", 4)
    resp = client.post("/transcribe-source", files=_files())
    assert resp.status_code == 422


def test_transcribe_source_503_when_model_unavailable(client, monkeypatch):
    _mock_source_pipeline(monkeypatch, error=asr.AsrUnavailableError("no model"))
    resp = client.post("/transcribe-source", files=_files())
    assert resp.status_code == 503


def test_transcribe_source_500_on_unexpected_failure(client, monkeypatch):
    _mock_source_pipeline(monkeypatch, error=RuntimeError("boom"))
    resp = client.post("/transcribe-source", files=_files())
    assert resp.status_code == 500


def test_transcribe_source_retries_without_vad_when_vad_drops_everything(monkeypatch):
    """Silero VAD can classify a whole music-heavy track as non-speech;
    transcribe_source retries with vad_filter off rather than return []."""
    calls = []

    class _Seg:
        def __init__(self, text):
            self.text, self.start, self.end = text, 0.0, 1.0
            self.avg_logprob, self.no_speech_prob = -0.3, 0.1

    def fake_run(_model, _wav, *, vad):
        calls.append(vad)
        return [] if vad else [_Seg("歌詞だよ。")]

    monkeypatch.setattr(asr, "_run_source_transcribe", fake_run)
    monkeypatch.setattr(asr, "_get_source_model", lambda: object())
    monkeypatch.setattr(config, "SOURCE_WHISPER_UNLOAD", False)

    out = asr.transcribe_source(Path("/tmp/fake.wav"))

    assert calls == [True, False]
    assert [s["text"] for s in out] == ["歌詞だよ。"]
