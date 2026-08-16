"""Real end-to-end alignment against a self-synthesized clip — no
copyrighted/checked-in audio needed (see docs/STATUS.md Phase 9, Milestone
2a in jp_sentence_splits for why: the app already runs VOICEVOX TTS on this
host). Skipped automatically when the MFA models or the VOICEVOX wrapper
aren't available, so the rest of the suite still runs anywhere.
"""
import httpx
import pytest

from app import aligner
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


@pytest.mark.skipif(not aligner.models_present(), reason="MFA models not downloaded")
def test_real_alignment_matches_expected_words():
    audio_bytes = _synthesize(TEST_SENTENCE)
    if audio_bytes is None:
        pytest.skip("VOICEVOX TTS wrapper not reachable at 127.0.0.1:8001")

    with transcoded_wav(audio_bytes) as wav_path:
        result = aligner.align(wav_path, TEST_SENTENCE)

    words = [w["text"] for w in result["words"] if w["text"] not in ("", "<eps>", "sil")]
    assert words == ["今日", "は", "ちょっと", "寒い", "です", "ね"]
    assert result["durationSeconds"] > 0

    chotto = next(w for w in result["words"] if w["text"] == "ちょっと")
    # っ (sokuon) is a held consonant — MFA marks it with a length diacritic.
    assert any("ː" in phone["text"] for phone in chotto["phones"])
