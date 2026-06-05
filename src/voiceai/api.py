from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from .diarization import diarize
from .emotion import classify_segments
from .output import build_result, build_text
from .transcribe import transcribe

app = FastAPI(
    title="voiceai",
    version="0.1.0",
)

@app.get("/health")
def health() -> dict:
    """Проверка живости сервиса."""
    return {"status": "ok"}

@app.post("/transcribe")
def transcribe_endpoint(
    file: UploadFile = File(..., description="аудиофайл с записью разговора"),
    base_url: str | None = Form(None, description="URL OpenAI-совместимого endpoint vLLM"),
    num_speakers: int = Form(2, description="ожидаемое число спикеров"),
    device: str = Form("auto", description="устройство диаризации/эмоций: auto/cpu/cuda"),
    detect_emotions: bool = Form(True, description="определять эмоции по сегментам"),
    response_format: str = Form("json", description="формат ответа: json или txt"),
):

    suffix = Path(file.filename or "audio").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        try:
            result = transcribe(tmp_path, language="ru", base_url=base_url)
        except Exception as e:  # ошибки обращения к vLLM-серверу
            raise HTTPException(
                status_code=502,
                detail=f"Ошибка обращения к vLLM ({type(e).__name__}): {e}",
            ) from e

        try:
            diarization = diarize(tmp_path, num_speakers=num_speakers, device=device)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка диаризации ({type(e).__name__}): {e}",
            ) from e

        # Эмоции необязательны: при сбое не валим весь запрос, просто без них.
        emotions = None
        if detect_emotions:
            try:
                emotions = classify_segments(tmp_path, result.segments, device=device)
            except Exception:
                emotions = None

        if response_format == "txt":
            return PlainTextResponse(build_text(result, diarization, emotions))
        return build_result(result, diarization, emotions)
    finally:
        os.unlink(tmp_path)

def run() -> None:
    import uvicorn

    uvicorn.run(
        "voiceai.api:app",
        host=os.environ.get("VOICEAI_HOST", "0.0.0.0"),
        port=int(os.environ.get("VOICEAI_PORT", "8080")),
    )

if __name__ == "__main__":
    run()
