"""Warm, in-process Japanese forced alignment via Montreal Forced Aligner's
online-alignment API.

Deliberately NOT implemented by shelling out to `mfa align`/`mfa align_one`
per request: measured on this host, that CLI path costs ~45-165s of fixed
per-invocation overhead (corpus/database setup, lexicon FST compilation,
worker pool spin-up) regardless of clip length — unusable for an interactive
"record, then analyze" loop. `montreal_forced_aligner.online.alignment
.align_utterance_online` is a lower-level API meant for exactly this case:
load the acoustic model + compiled lexicon + tokenizer ONCE (measured
~40s, dominated by lexicon FST compilation) and reuse them across many
single-utterance alignments (measured ~1.5s each afterward). See
docs/STATUS.md Phase 9, Milestone 2a in jp_sentence_splits for the full
investigation.
"""
import re
import threading
from pathlib import Path
from typing import TypedDict

from kalpy.feat.cmvn import CmvnComputer
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.utterance import Segment
from kalpy.utterance import Utterance as KalpyUtterance
from montreal_forced_aligner.models import AcousticModel, DictionaryModel
from montreal_forced_aligner.online.alignment import align_utterance_online
from montreal_forced_aligner.tokenization.spacy import generate_language_tokenizer

from app import config


class PhoneInterval(TypedDict):
    start: float
    end: float
    text: str


class WordInterval(TypedDict):
    start: float
    end: float
    text: str
    phones: list[PhoneInterval]


class AlignmentResult(TypedDict):
    durationSeconds: float
    words: list[WordInterval]


class AlignerUnavailableError(RuntimeError):
    pass


class _AlignerState:
    def __init__(self) -> None:
        self.acoustic_model = AcousticModel(config.ACOUSTIC_MODEL_PATH)
        dictionary = DictionaryModel(config.DICTIONARY_PATH)
        params = self.acoustic_model.parameters
        self.lexicon_compiler = LexiconCompiler(
            disambiguation=False,
            silence_probability=params["silence_probability"],
            initial_silence_probability=params["initial_silence_probability"],
            final_silence_correction=params["final_silence_correction"],
            final_non_silence_correction=params["final_non_silence_correction"],
            silence_phone=params["optional_silence_phone"],
            oov_phone=params["oov_phone"],
            position_dependent_phones=params["position_dependent_phones"],
            phones=params["non_silence_phones"],
            ignore_case=True,
        )
        self.lexicon_compiler.load_pronunciations(dictionary.path)
        self.lexicon_compiler.create_fsts()
        self.tokenizer = generate_language_tokenizer(self.acoustic_model.language)


# Guards both lazy `_state` construction and `align()` itself — kalpy's
# aligner/lexicon aren't documented as safe for concurrent use, and this is
# a single-user personal service where serialized ~1-3s alignments cost
# nothing real.
_lock = threading.Lock()
_state: _AlignerState | None = None


def models_present() -> bool:
    return config.ACOUSTIC_MODEL_PATH.is_file() and config.DICTIONARY_PATH.is_file()


def is_loaded() -> bool:
    return _state is not None


def _get_state() -> _AlignerState:
    global _state
    if _state is None:
        if not models_present():
            raise AlignerUnavailableError(
                f"Acoustic model or dictionary not found (expected "
                f"{config.ACOUSTIC_MODEL_PATH} and {config.DICTIONARY_PATH}). "
                f"Run `mfa model download acoustic japanese_mfa` and "
                f"`mfa model download dictionary japanese_mfa`."
            )
        _state = _AlignerState()
    return _state


_PHONE_VARIANT_SUFFIX = re.compile(r"\(\d+\)$")


def _clean_phone_label(label: str) -> str:
    # kalpy labels position-dependent phone variants like "tɕ(46)" — the
    # numeric suffix is an internal disambiguation id, not part of the phone.
    return _PHONE_VARIANT_SUFFIX.sub("", label)


def align(wav_path: Path, transcript: str) -> AlignmentResult:
    """Aligns one short utterance against `transcript`. Blocking/CPU-bound —
    callers on an event loop should run this via a thread (e.g.
    `asyncio.to_thread`)."""
    with _lock:
        state = _get_state()
        segment = Segment(wav_path, 0, None, 0)
        utterance = KalpyUtterance(segment, transcript)
        utterance.generate_mfccs(state.acoustic_model.mfcc_computer)
        cmvn = CmvnComputer().compute_cmvn_from_features([utterance.mfccs])
        utterance.apply_cmvn(cmvn)
        ctm = align_utterance_online(
            state.acoustic_model,
            utterance,
            state.lexicon_compiler,
            tokenizer=state.tokenizer,
        )

    words: list[WordInterval] = []
    duration = 0.0
    for word_interval in ctm.word_intervals:
        duration = max(duration, float(word_interval.end))
        words.append(
            {
                "start": float(word_interval.begin),
                "end": float(word_interval.end),
                "text": word_interval.label,
                "phones": [
                    {
                        "start": float(phone.begin),
                        "end": float(phone.end),
                        "text": _clean_phone_label(phone.label),
                    }
                    for phone in word_interval.phones
                ],
            }
        )
    return {"durationSeconds": duration, "words": words}
