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

API_HOST = os.environ.get("ANALYSIS_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("ANALYSIS_API_PORT", "8002"))

MAX_TRANSCRIPT_LENGTH = int(os.environ.get("ANALYSIS_MAX_TRANSCRIPT_LENGTH", "200"))
# A shadowing clip is one sentence — long uploads are almost certainly a
# mistake, not a legitimate use case.
MAX_AUDIO_BYTES = int(os.environ.get("ANALYSIS_MAX_AUDIO_BYTES", str(20 * 1024 * 1024)))
