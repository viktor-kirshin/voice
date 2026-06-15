from __future__ import annotations

import os
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, RedirectResponse

from .audio import load_waveform
from .emotion import classify_segments
from .output import build_result, build_text
from .transcribe import transcribe

app = FastAPI(
    title="voiceai",
    version="0.1.0",
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Корень → интерактивная документация (Swagger UI)."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    """Проверка живости сервиса."""
    return {"status": "ok"}


@app.post("/transcribe")
def transcribe_endpoint(
    file: UploadFile = File(..., description="аудиофайл с записью разговора"),
    base_url: str | None = Form(None, description="URL OpenAI-совместимого endpoint vLLM"),
    prompt: str | None = Form(None, description="подсказка Whisper: имена/термины для точности"),
    detect_emotions: bool = Form(True, description="определять эмоции по сегментам"),
    response_format: str = Form("json", description="формат ответа: json или txt"),
):
    if response_format not in ("json", "txt"):
        raise HTTPException(
            status_code=422, detail="response_format должен быть 'json' или 'txt'"
        )

    suffix = Path(file.filename or "audio").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        try:
            result = transcribe(tmp_path, language="ru", base_url=base_url,
                                prompt=prompt)
        except Exception as e:  # ошибки обращения к vLLM-серверу
            raise HTTPException(
                status_code=502,
                detail=f"Ошибка обращения к vLLM ({type(e).__name__}): {e}",
            ) from e

        # эмоции необязательны: при сбое не валим запрос, просто без них
        emotions = None
        if detect_emotions:
            try:
                waveform, sr = load_waveform(tmp_path)
                emotions = classify_segments(waveform, sr, result.segments)
            except Exception:
                emotions = None

        if response_format == "txt":
            return PlainTextResponse(build_text(result, emotions))
        return build_result(result, emotions)
    finally:
        os.unlink(tmp_path)


def run() -> None:
    uvicorn.run(
        "voiceai.api:app",
        host=os.environ.get("VOICEAI_HOST", "0.0.0.0"),
        port=int(os.environ.get("VOICEAI_PORT", "8080")),
    )


if __name__ == "__main__":
    run()
