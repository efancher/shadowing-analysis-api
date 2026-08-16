"""ffmpeg-based transcoding: arbitrary browser-recorded audio (webm/opus,
mp4, wav, ...) -> 16 kHz mono WAV, the format kalpy's feature extraction
expects. ffmpeg is a system binary already present on this host (not a new
dependency introduced by this service)."""
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class AudioTranscodeError(RuntimeError):
    pass


@contextmanager
def transcoded_wav(audio_bytes: bytes) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="shadowing-analysis-") as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "input"
        output_path = tmp_path / "audio.wav"
        input_path.write_bytes(audio_bytes)
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(input_path),
                "-ar", "16000",
                "-ac", "1",
                str(output_path),
            ],
            capture_output=True,
        )
        if result.returncode != 0 or not output_path.is_file():
            stderr = result.stderr.decode("utf-8", "replace")
            raise AudioTranscodeError(stderr[-2000:])
        yield output_path
