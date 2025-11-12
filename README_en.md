<div style="text-align: center;">
<h1>Kokoro TTS Service</h1>
<p><a href="README.md">中文</a> | <a href="README_en.md">English</a></p>
</div>

This project wraps the inference of the `hexgrad/Kokoro-82M-v1.1-zh` model in a lightweight FastAPI service, providing a `/text2audio` endpoint for TTS usage in [Legado 3.0](https://github.com/gedoor/legado).

## Prerequisites

- Python 3.12+
- Kokoro 1.1 Chinese weights and voice files placed in the `model/` directory
- Access token configured via the `KOKORO_API_TOKEN` environment variable (required for API calls)

Optional but recommended:

- CUDA-capable GPU when running with `device=cuda`

## Environment Setup

```bash
# Create virtual environment and install dependencies
uv lock
uv sync
```

## Running the Service

```powershell
# Execute from the project root directory
uv run uvicorn tts_service:app --host 0.0.0.0 --port 3236

# Example terminal output
# INFO kokoro_tts: Kokoro model loaded on cuda
# INFO kokoro_tts.monitor: cpu=5.12% rss=1024.00MB ...
```

After successful startup, you can access:

- `http://127.0.0.1:3236/health` — View if the model is loaded and current device
- `http://127.0.0.1:3236/text2audio?token=<token>&tex=卧槽小米怎么这么坏啊` — Quick check of audio output

Main endpoints:

- `GET /text2audio`
- `POST /text2audio`
- `GET /health`

Important query/form parameters:

| Name         | Type    | Description                                                      |
|--------------|---------|------------------------------------------------------------------|
| `tex`/`text` | string  | Text to synthesize. Double URL-encoded values will be auto-decoded. |
| `speakSpeed` | float   | Preferred speech speed, range 5-50.                              |
| `spd`        | float   | Speed variant. Converted to the same 5-50 range.                 |
| `voice`      | string  | Optional Kokoro voice profile (e.g. `zf_001`).                   |
| `device`     | string  | One of `cpu`, `cuda` or `auto`. Defaults to `KOKORO_DEVICE` env var or `auto`. |
| `token`      | string  | Access token can be passed via query parameter when client cannot set headers. |

Response is returned as `audio/wav` stream with the following headers:

- `X-RTF`: Real-time factor of synthesis process
- `X-Sample-Rate`: Audio sample rate (24 kHz)
- `X-Device`: Actual device used for inference
- Calls to the API must carry a valid access token, which can be provided via header (default `X-API-Token`) or by appending `?token=<token>` to the URL.

If CUDA is requested but unavailable, or if a CUDA runtime error occurs during inference, the service automatically retries with CPU.

## Monitoring and Logging

- Runtime logs are written to `logs/tts_service.log`. Files are rotated when reaching 5MB, keeping three backups to prevent unlimited growth. Can be adjusted via environment variables (see below).
- Lightweight resource monitor logs CPU usage and (when CUDA is available) GPU memory every 10 seconds. Sampling frequency can be changed via `KOKORO_MONITOR_INTERVAL`.

## Configuration Environment Variables

| Variable Name               | Default   | Purpose                                                       |
|-----------------------------|-----------|---------------------------------------------------------------|
| `KOKORO_DEVICE`             | `auto`    | Preference for `cpu`, `cuda` or `auto` at startup             |
| `KOKORO_MONITOR_INTERVAL`   | `10`      | Resource monitor sampling interval (seconds)                  |
| `KOKORO_LOG_MAX_BYTES`      | `5000000` | Maximum size before log file rotation                         |
| `KOKORO_LOG_BACKUP_COUNT`   | `3`       | Number of rotated log files to retain                         |
| `KOKORO_API_TOKEN`          | _(not set)_ | List of tokens allowed to access the service                  |
| `KOKORO_TOKEN_HEADER`       | `X-API-Token` | Name of request header carrying the token                     |

## Example Requests

### PowerShell

```powershell
Invoke-WebRequest -Method POST `
  -Uri "http://127.0.0.1:3236/text2audio" `
  -Headers @{ "X-API-Token" = "<your-token>" } `
  -Body "tex=卧槽小米怎么这么坏啊&voice=zf_001&device=cuda" `
  -ContentType "application/x-www-form-urlencoded" `
  -OutFile "demo.wav"
```

### Bash/zsh

```bash
# Use curl to call /text2audio via HTTP POST and save output as demo.wav
curl -X POST 'http://127.0.0.1:3236/text2audio' \
  -H 'X-API-Token: <your-token>' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode 'tex=卧槽小米怎么这么坏啊' \
  --data 'voice=zf_001&device=cuda' \
  --output demo.wav
```

### URL

```text
http://127.0.0.1:3236/text2audio?token=<your-token>&tex=卧槽小米怎么这么坏啊
```

Check response headers to see the values of `X-RTF` and `X-Device`.

## Legado (Reading 3.0) Configuration Example

To add a custom configuration in Legado 3.0 App's "Read Aloud Engine", you can refer to the following JSON and paste the source:

```json
{
  "concurrentRate": "",
  "contentType": "audio/wav",
  "enabledCookieJar": false,
  "header": "Content-Type: application/x-www-form-urlencoded",
  "id": -100,
  "lastUpdateTime": 1762950914174,
  "loginCheckJs": "",
  "loginUi": "",
  "loginUrl": "",
  "name": "Kokoro",
  "url": "http://<server IP or domain>:3236/text2audio?token=<your-token>,{\"method\":\"POST\",\"body\":\"tex={{java.encodeURI(java.encodeURI(speakText))}}&speakSpeed={{speakSpeed}}&voice=zf_001\"}"
}
```

Notes:

- `speakSpeed` corresponds to the "speech speed" slider in Legado, the server automatically maps to the 5-50 range
- The `voice` parameter can be replaced with other available voice line names in the `voices/` directory (requires corresponding language)