"""Japanese ASR via faster-whisper.

Two uses, two models:

  - `transcribe()` — a secondary *diagnostic* signal (jp_sentence_splits
    Phase 9 Milestone 7): the caller already knows the expected transcript,
    so this only flags "possible pronunciation difference" hints. Small
    `base` model (~270 MB RSS int8), chosen because `aligner.py` already
    uses ~2.4 GB warm and this host had ~1.5 GB free when it was built.

  - `transcribe_source()` — a *full source transcript* for the mining
    pipeline (v2, `POST /transcribe-source`), where the transcript is the
    product. Larger `SOURCE_WHISPER_MODEL` (default `small`), timed
    segments, no prompt. Separate lazily-loaded instance so the diagnostic
    path keeps its tiny footprint.

Both load lazily on first use (keep `systemd` startup fast).
"""
import gc
import threading
from pathlib import Path

from faster_whisper import WhisperModel

from app import config


class AsrUnavailableError(RuntimeError):
    pass


_lock = threading.Lock()
_model: WhisperModel | None = None
_source_lock = threading.Lock()
_source_model: WhisperModel | None = None


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


def _get_source_model() -> WhisperModel:
    global _source_model
    if _source_model is None:
        try:
            _source_model = WhisperModel(
                config.SOURCE_WHISPER_MODEL, device="cpu", compute_type="int8"
            )
        except Exception as exc:
            raise AsrUnavailableError(f"Could not load Whisper model: {exc}") from exc
    return _source_model


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


def transcribe_source(wav_path: Path) -> list[dict]:
    """Transcribe a full recording into timed segments:
    `[{text, startMs, endMs, avgLogprob, noSpeechProb}]`.

    `vad_filter` skips music/silence (intros, BGM stretches);
    `condition_on_previous_text=False` curbs Whisper's repetition-loop
    hallucination on long audio. Blocking/CPU-bound — run via a thread.
    """
    global _source_model
    with _source_lock:
        model = _get_source_model()
        try:
            segments = _run_source_transcribe(model, wav_path, vad=True)
            if not segments:
                # Silero VAD sometimes classifies an entire music-heavy track
                # (a song, dense BGM) as non-speech and drops everything —
                # losing real speech is worse than a stray hallucinated
                # segment the reviewer can delete.
                segments = _run_source_transcribe(model, wav_path, vad=False)
            return [
                {
                    "text": seg.text.strip(),
                    "startMs": max(0, round(seg.start * 1000)),
                    "endMs": max(round(seg.end * 1000), round(seg.start * 1000) + 1),
                    "avgLogprob": seg.avg_logprob,
                    "noSpeechProb": seg.no_speech_prob,
                }
                for seg in segments
                if seg.text.strip()
            ]
        finally:
            if config.SOURCE_WHISPER_UNLOAD:
                # Drops the Python object; CTranslate2 keeps some of its
                # allocation pool for the process lifetime, so this reclaims
                # roughly half of the ~1.8 GB, not all of it. Restart the
                # service to fully reclaim.
                del model
                _source_model = None
                gc.collect()


def _run_source_transcribe(model: WhisperModel, wav_path: Path, *, vad: bool):
    segments, _info = model.transcribe(
        str(wav_path),
        language="ja",
        vad_filter=vad,
        condition_on_previous_text=False,
    )
    return list(segments)
