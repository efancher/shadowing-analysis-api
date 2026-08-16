import asyncio
import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import aligner, audio, config

logger = logging.getLogger("shadowing_analysis_api")

app = FastAPI(title="Shadowing Analysis API")

# The jp_sentence_splits frontend only — this service is also restricted at
# the network layer (Tailscale-tailnet-only), CORS is an extra layer, not
# the only one.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mfa": {
            "modelsPresent": aligner.models_present(),
            "loaded": aligner.is_loaded(),
        },
    }


@app.post("/align")
async def align_audio(
    audio_file: UploadFile = File(..., alias="audio"),
    transcript: str = Form(...),
):
    text = transcript.strip()
    if not text:
        raise HTTPException(status_code=422, detail="transcript must not be blank")
    if len(text) > config.MAX_TRANSCRIPT_LENGTH:
        raise HTTPException(status_code=422, detail="transcript too long")

    data = await audio_file.read()
    if not data:
        raise HTTPException(status_code=422, detail="audio must not be empty")
    if len(data) > config.MAX_AUDIO_BYTES:
        raise HTTPException(status_code=422, detail="audio too large")

    try:
        with audio.transcoded_wav(data) as wav_path:
            try:
                return await asyncio.to_thread(aligner.align, wav_path, text)
            except aligner.AlignerUnavailableError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except HTTPException:
                raise
            except Exception as exc:
                logger.warning("Alignment failed: %s", exc)
                raise HTTPException(status_code=500, detail="Alignment failed") from exc
    except audio.AudioTranscodeError as exc:
        raise HTTPException(
            status_code=422, detail=f"Could not decode audio: {exc}"
        ) from exc
