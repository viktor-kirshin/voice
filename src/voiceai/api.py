from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse

from .emotion import load_emotion_model
from .output import build_result
from .transcribe import transcribe


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Прогрев модели эмоций один раз при старте сервера.

    Грузится здесь и переиспользуется всеми запросами (в app.state). Если веса
    не скачались — сервис падает сразу на старте (fail-fast), а не отдаёт 500
    в середине обработки.
    """
    device = os.environ.get("VOICEAI_DEVICE")
    app.state.emotion_model = load_emotion_model(device)
    yield
    app.state.emotion_model = None


app = FastAPI(
    title="voiceai",
    version="0.1.0",
    lifespan=lifespan,
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
    request: Request,
    file: UploadFile = File(..., description="аудиофайл с записью разговора"),
    base_url: str | None = Form(None, description="URL OpenAI-совместимого endpoint vLLM"),
    prompt: str | None = Form(None, description="подсказка Whisper: имена/термины для точности"),
    detect_emotions: bool = Form(True, description="определять эмоцию записи"),
):
    """Распознаёт речь (Whisper@vLLM) и эмоцию записи. Ответ — JSON, язык русский."""
    suffix = Path(file.filename or "audio").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        try:
            result = transcribe(tmp_path, language="ru", base_url=base_url, prompt=prompt)
        except Exception as e:  # ошибки обращения к vLLM-серверу
            raise HTTPException(
                status_code=502,
                detail=f"Ошибка обращения к vLLM ({type(e).__name__}): {e}",
            ) from e

        # эмоция необязательна: при сбое не валим запрос, просто без неё
        emotion, scores = None, None
        if detect_emotions:
            try:
                emotion, scores = request.app.state.emotion_model.predict(tmp_path)
            except Exception:
                emotion, scores = None, None

        return build_result(result, emotion, scores)
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
