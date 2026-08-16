"""Real end-to-end ASR against a self-synthesized clip — same rationale as
test_integration_alignment.py: no checked-in audio needed, since VOICEVOX
TTS is already running on this host. Skipped automatically when reachable
prerequisites aren't met.
"""
import httpx
import pytest

from app import asr
from app.audio import transcoded_wav

pytestmark = pytest.mark.slow

VOICEVOX_TTS_URL = "http://127.0.0.1:8001/tts"
TEST_SENTENCE = "今日はちょっと寒いですね"


def _synthesize(text: str) -> bytes | None:
    try:
        resp = httpx.post(VOICEVOX_TTS_URL, json={"text": text, "speaker": 2}, timeout=30)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


def test_real_transcription_recognizes_the_sentence():
    audio_bytes = _synthesize(TEST_SENTENCE)
    if audio_bytes is None:
        pytest.skip("VOICEVOX TTS wrapper not reachable at 127.0.0.1:8001")

    with transcoded_wav(audio_bytes) as wav_path:
        text = asr.transcribe(wav_path, prompt=TEST_SENTENCE)

    assert text == TEST_SENTENCE
