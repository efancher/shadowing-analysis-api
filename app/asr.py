"""Japanese ASR via faster-whisper — a secondary diagnostic signal only,
never ground truth (docs/STATUS.md Phase 9, Milestone 7 in
jp_sentence_splits: the caller already knows the expected transcript, so
this is used to flag "possible pronunciation difference" hints, not to
grade correctness).

Resource note: measured on this host, the `base` model at int8 quantization
uses ~270 MB RSS once loaded (not the ~2 GB the `small` model would have
needed) — chosen specifically because the alignment service (`aligner.py`)
already uses ~2.4 GB RSS once warm, and this host had only ~1.5 GB
genuinely free when this was built. Loads lazily on first use, same
reasoning as `aligner.py` (keep `systemd` startup fast).
"""
import threading
from pathlib import Path

from faster_whisper import WhisperModel

from app import config


class AsrUnavailableError(RuntimeError):
    pass


_lock = threading.Lock()
_model: WhisperModel | None = None


def is_loaded() -> bool:
    return _model is not None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        try:
            _model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
        except Exception as exc:
            raise AsrUnavailableError(f"Could not load Whisper model: {exc}") from exc
    return _model


def transcribe(wav_path: Path, prompt: str | None = None) -> str:
    """Transcribes one short utterance. Blocking/CPU-bound — callers on an
    event loop should run this via a thread (e.g. `asyncio.to_thread`)."""
    with _lock:
        model = _get_model()
        segments, _info = model.transcribe(
            str(wav_path),
            language="ja",
            initial_prompt=prompt,
        )
        return "".join(segment.text for segment in segments).strip()
