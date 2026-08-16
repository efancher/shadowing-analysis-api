# shadowing-analysis-api

A small Japanese forced-alignment API: given a short audio clip and its
known transcript, returns word- and phone-level time boundaries (via
[Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/)'s
`japanese_mfa` acoustic model + dictionary). Built for
[jp_sentence_splits](../jp_sentence_splits)' shadowing pronunciation-feedback
feature (see that repo's `docs/STATUS.md`, Phase 9) — the exact Japanese
sentence and both a native reference recording and a learner recording are
known ahead of time, so alignment (not open-vocabulary ASR) is the right
tool.

Like `voicevox-tts-api`, this service is intended to stay **tailnet-only**
(Tailscale `serve`, never `funnel`/public). It has no authentication of its
own — that's fine only because network-layer access is already restricted.
Do not expose it publicly without adding auth first.

## Why this needs a conda environment, not just a venv

`montreal-forced-aligner`'s Kaldi bindings (`kalpy`) and `pynini` (OpenFst)
are not available as portable pip wheels on Linux — MFA's own docs are
explicit that conda/mamba is the supported install path. This app's own
Python dependencies (FastAPI, uvicorn, etc., see `requirements.txt`) are
installed *into that same conda environment* rather than a separate venv, so
one Python interpreter has both.

## Why alignment is done in-process, not by shelling out to `mfa align`

Measured on this host: `mfa align` (full corpus pipeline) costs ~155-165s
per invocation regardless of clip length (corpus/database setup, lexicon FST
compilation, multiprocessing worker spin-up); `mfa align_one` (single-file,
no corpus database) is faster but still ~45-50s per invocation. Both are far
too slow for an interactive "record, then see feedback" loop.
`montreal_forced_aligner.online.alignment.align_utterance_online` is a
lower-level API meant for exactly this case — load the acoustic model,
compiled lexicon, and tokenizer **once**, then reuse them across many
single-utterance alignments. Measured: ~40s one-time load (dominated by
lexicon FST compilation), then **~1-3s per alignment** after that. This app
(`app/aligner.py`) loads lazily on the first `/align` request and keeps the
loaded state in memory for the life of the process — the systemd service
below is expected to stay running, not restart per request.

## Install

```bash
# 1. Miniforge (conda + mamba), if not already installed:
curl -fsSL -o /tmp/miniforge.sh \
  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash /tmp/miniforge.sh -b -p ~/miniforge3

# 2. The `mfa` environment:
~/miniforge3/bin/mamba create -n mfa -c conda-forge montreal-forced-aligner -y
~/miniforge3/envs/mfa/bin/mfa model download acoustic japanese_mfa
~/miniforge3/envs/mfa/bin/mfa model download dictionary japanese_mfa

# 3. This app's own dependencies, into the same environment:
~/miniforge3/envs/mfa/bin/pip install -r requirements-dev.txt
```

Also requires `ffmpeg`/`ffprobe` on `PATH` (used to transcode uploaded audio
to 16 kHz mono WAV before alignment) — a system package, not installed by
the steps above.

## Run

Directly:

```bash
~/miniforge3/envs/mfa/bin/uvicorn app.main:app --host 127.0.0.1 --port 8002
```

Or via the included systemd user unit (recommended — keeps the loaded
models warm across the service's lifetime; restarts on failure):

```bash
mkdir -p ~/.config/systemd/user
cp shadowing-analysis-api.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now shadowing-analysis-api
```

This binds only to `127.0.0.1:8002` — do not change the host without
deliberately deciding to expose it. For boot persistence (the user service
would otherwise stop when your last session ends), enable linger once
(needs root, so this is not scripted here):

```bash
sudo loginctl enable-linger "$USER"
```

### Exposing over Tailscale (tailnet-only)

```bash
tailscale serve --bg --set-path /shadowing-analysis http://127.0.0.1:8002
```

Requires either root or being the tailscale operator
(`sudo tailscale set --operator=$USER`, one-time). Verify with
`tailscale serve status` — this should coexist with any other path already
mounted (e.g. `voicevox-tts-api` at `/`), not replace it. No `funnel` —
tailnet-only, matching this whole app ecosystem's privacy posture.

## Configuration

All optional, via environment variables:

| Variable                       | Default                                                    | Meaning                          |
|---------------------------------|--------------------------------------------------------------|-----------------------------------|
| `MFA_ROOT`                     | `~/Documents/MFA`                                            | MFA's own data directory          |
| `MFA_ACOUSTIC_MODEL_PATH`      | `$MFA_ROOT/pretrained_models/acoustic/japanese_mfa.zip`      | Acoustic model path                |
| `MFA_DICTIONARY_PATH`          | `$MFA_ROOT/pretrained_models/dictionary/japanese_mfa.dict`   | Dictionary path                    |
| `ANALYSIS_API_HOST`            | `127.0.0.1`                                                   | Documented default host            |
| `ANALYSIS_API_PORT`            | `8002`                                                        | Documented default port            |
| `ANALYSIS_MAX_TRANSCRIPT_LENGTH` | `200`                                                       | Max characters accepted by `/align`|
| `ANALYSIS_MAX_AUDIO_BYTES`     | `20971520` (20 MB)                                            | Max upload size for `/align`       |
| `ANALYSIS_ALLOWED_ORIGINS`     | `https://efancher.github.io,http://localhost:5173,http://127.0.0.1:5173` | Comma-separated CORS allow-list |

## API

### `GET /health`

```json
{ "status": "ok", "mfa": { "modelsPresent": true, "loaded": false } }
```

`loaded` is `false` until the first `/align` request triggers the one-time
model/lexicon load; `modelsPresent` just checks the files exist on disk.

### `POST /align`

Multipart form: `audio` (a file — webm/opus, mp4, wav, anything `ffmpeg` can
read) and `transcript` (the known Japanese sentence text, plain string, no
pre-tokenization needed — MFA's own Japanese tokenizer handles raw text).

```bash
curl -X POST http://127.0.0.1:8002/align \
  -F "audio=@clip.wav" \
  -F "transcript=今日はちょっと寒いですね" 
```

Returns word-level intervals, each with nested phone-level intervals:

```json
{
  "durationSeconds": 1.696,
  "words": [
    {
      "start": 0.5,
      "end": 0.84,
      "text": "ちょっと",
      "phones": [
        { "start": 0.5, "end": 0.6, "text": "tɕ" },
        { "start": 0.6, "end": 0.69, "text": "o" },
        { "start": 0.69, "end": 0.79, "text": "tː" },
        { "start": 0.79, "end": 0.84, "text": "o" }
      ]
    }
  ]
}
```

Silence at the start/end of the clip appears as a word interval with
`text: "<eps>"` (phone `"sil"`) — not filtered out, since a later consumer
may care about leading/trailing pause length. No persistent cache — each
call recomputes (the ~1-3s warm cost is cheap enough that caching isn't
worth the complexity yet; revisit if that changes).

Errors: `422` for a blank/too-long transcript, missing/empty/oversized
audio, or audio `ffmpeg` can't decode; `503` if the MFA models aren't
downloaded; `500` for an unexpected alignment failure.

## Tests

```bash
~/miniforge3/envs/mfa/bin/python3 -m pytest -m "not slow"   # fast, no real models/audio needed
~/miniforge3/envs/mfa/bin/python3 -m pytest -m slow         # real end-to-end, ~45s
~/miniforge3/envs/mfa/bin/python3 -m pytest                 # everything
```

The `slow` test synthesizes its own test clip via the `voicevox-tts-api`
wrapper already running on this host (`127.0.0.1:8001`) rather than using a
checked-in audio fixture, and is skipped automatically if that's
unreachable or the MFA models aren't downloaded.

## Notes

- No authentication is implemented; this is safe only because the service
  is bound to `127.0.0.1` and only reachable beyond that via the
  Tailscale-tailnet-only `serve` mount above. Do not put this behind a
  public port without adding auth first.
- CORS is allow-listed to the `jp_sentence_splits` frontend origin(s) only
  (`app/config.py`'s `ANALYSIS_ALLOWED_ORIGINS`, default `https://efancher
  .github.io` plus local dev origins) — not a wildcard, even though the
  service is also restricted at the network layer.
