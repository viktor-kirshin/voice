from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from .diarization import diarize
from .output import build_result, build_text
from .transcribe import transcribe

app = FastAPI(
    title="voiceai",
    description=(
        "Транскрибация диалогов: Whisper на vLLM (OpenAI endpoint) + "
        "диаризация спикеров pyannote + пометка нецензурной лексики."
    ),
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    """Проверка живости сервиса."""
    return {"status": "ok"}


@app.post("/transcribe")
def transcribe_endpoint(
    file: UploadFile = File(..., description="аудиофайл с записью разговора"),
    language: str | None = Form(None, description="код языка (ru/en/...) или auto"),
    model: str = Form("openai/whisper-large-v3", description="id модели на vLLM"),
    base_url: str | None = Form(None, description="URL OpenAI-совместимого endpoint vLLM"),
    api_key: str | None = Form(None, description="ключ endpoint (vLLM игнорирует)"),
    num_speakers: int = Form(2, description="ожидаемое число спикеров"),
    device: str = Form("auto", description="устройство диаризации: auto/cpu/cuda"),
    response_format: str = Form("json", description="формат ответа: json или txt"),
):
    """Распознаёт речь (Whisper@vLLM) и размечает спикеров (pyannote).

    Возвращает JSON со структурой сегментов либо готовый текст (`txt`).
    Эндпоинт синхронный — FastAPI исполняет его в пуле потоков, поэтому
    тяжёлая модель не блокирует event loop.
    """
    # Загрузку сохраняем во временный файл: и vLLM-клиент, и ffmpeg для
    # диаризации работают с путём к файлу.
    suffix = Path(file.filename or "audio").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        lang = None if language in (None, "", "auto") else language

        try:
            result = transcribe(tmp_path, model=model, language=lang,
                                base_url=base_url, api_key=api_key)
        except Exception as e:  # ошибки обращения к vLLM-серверу
            raise HTTPException(
                status_code=502,
                detail=f"Ошибка обращения к vLLM ({type(e).__name__}): {e}",
            ) from e

        try:
            diarization = diarize(tmp_path, num_speakers=num_speakers, device=device)
        except Exception as e:  # ошибки диаризации (токен HF, ffmpeg, ...)
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка диаризации ({type(e).__name__}): {e}",
            ) from e

        if response_format == "txt":
            return PlainTextResponse(build_text(result, diarization))
        return build_result(result, diarization)
    finally:
        os.unlink(tmp_path)


def run() -> None:
    """Точка входа `voiceai`: поднимает HTTP-сервер uvicorn."""
    import uvicorn

    uvicorn.run(
        "voiceai.api:app",
        host=os.environ.get("VOICEAI_HOST", "0.0.0.0"),
        port=int(os.environ.get("VOICEAI_PORT", "8080")),
    )


if __name__ == "__main__":
    run()
