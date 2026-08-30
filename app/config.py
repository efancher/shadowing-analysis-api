import os
from pathlib import Path

MFA_ROOT = Path(os.environ.get("MFA_ROOT", str(Path.home() / "Documents/MFA")))
ACOUSTIC_MODEL_PATH = Path(
    os.environ.get(
        "MFA_ACOUSTIC_MODEL_PATH",
        str(MFA_ROOT / "pretrained_models/acoustic/japanese_mfa.zip"),
    )
)
DICTIONARY_PATH = Path(
    os.environ.get(
        "MFA_DICTIONARY_PATH",
        str(MFA_ROOT / "pretrained_models/dictionary/japanese_mfa.dict"),
    )
)
# Small, hand-maintained additions layered on top of the pretrained
# japanese_mfa dictionary for words it has no entry for — chiefly casual
# contractions (e.g. 足んねえ, the colloquial negative of 足りない) that are
# common in real spoken/subtitled Japanese but aren't dictionary headwords.
# Loaded second, after DICTIONARY_PATH (see app/aligner.py), so an entry
# here can't shadow a real pretrained one. Optional: missing file is a
# no-op, not an error.
SUPPLEMENTARY_DICTIONARY_PATH = Path(
    os.environ.get(
        "MFA_SUPPLEMENTARY_DICTIONARY_PATH",
        str(Path(__file__).resolve().parent / "data" / "supplementary_dictionary.dict"),
    )
)

API_HOST = os.environ.get("ANALYSIS_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("ANALYSIS_API_PORT", "8002"))

MAX_TRANSCRIPT_LENGTH = int(os.environ.get("ANALYSIS_MAX_TRANSCRIPT_LENGTH", "200"))
# A shadowing clip is one sentence — long uploads are almost certainly a
# mistake, not a legitimate use case.
MAX_AUDIO_BYTES = int(os.environ.get("ANALYSIS_MAX_AUDIO_BYTES", str(20 * 1024 * 1024)))

# faster-whisper model size — "base" (int8), not "small": measured ~270 MB
# RSS vs. an estimated ~2 GB for "small", and this host had only ~1.5 GB
# genuinely free when this was added (the alignment service alone already
# uses ~2.4 GB warm). See app/asr.py.
WHISPER_MODEL = os.environ.get("ANALYSIS_WHISPER_MODEL", "base")

# A larger model for full-source transcription (jp_sentence_splits mining
# pipeline v2, POST /transcribe-source) — there the transcript IS the
# product, so kanji/punctuation accuracy matters, unlike the diagnostic ASR
# above. Lazily loaded on first mining run and kept resident. Set to "base"
# on a tighter box; "medium" if there's RAM headroom.
SOURCE_WHISPER_MODEL = os.environ.get("ANALYSIS_SOURCE_WHISPER_MODEL", "small")
# A mined source is minutes long, not one sentence — its own, larger cap.
MAX_SOURCE_AUDIO_BYTES = int(
    os.environ.get("ANALYSIS_MAX_SOURCE_AUDIO_BYTES", str(60 * 1024 * 1024))
)

# The jp_sentence_splits frontend (deployed origin + local dev). Safe to
# allow-list explicitly rather than wildcard — this is still only reachable
# at all over the Tailscale tailnet, CORS is just an extra layer on top.
ALLOWED_ORIGINS = os.environ.get(
    "ANALYSIS_ALLOWED_ORIGINS",
    "https://efancher.github.io,http://localhost:5173,http://127.0.0.1:5173",
).split(",")
