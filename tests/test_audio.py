import io
import wave

import pytest

from app.audio import AudioTranscodeError, transcoded_wav


def _tiny_wav_bytes(sample_rate: int = 44_100, duration_seconds: float = 0.1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * int(sample_rate * duration_seconds) * 2)
    return buffer.getvalue()


def test_transcodes_to_16k_mono_wav():
    with transcoded_wav(_tiny_wav_bytes()) as wav_path:
        assert wav_path.is_file()
        with wave.open(str(wav_path), "rb") as out:
            assert out.getframerate() == 16_000
            assert out.getnchannels() == 1


def test_raises_on_unreadable_audio():
    with pytest.raises(AudioTranscodeError):
        with transcoded_wav(b"this is not audio data"):
            pass
