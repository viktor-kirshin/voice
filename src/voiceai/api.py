from __future__ import annotations

import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse, RedirectResponse

from .audio import load_waveform
from .diarization import load_pipeline, run_diarization
from .emotion import classify_segments
from .output import build_result, build_text
from .transcribe import transcribe

_diarization_lock = threading.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):

    device = os.environ.get("VOICEAI_DEVICE", "auto")
    app.state.diarization_pipeline = load_pipeline(device=device)
    yield
    app.state.diarization_pipeline = None


app = FastAPI(
    title="voiceai",
    version="0.1.0",
    lifespan=lifespan,
)

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:

    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:

    return {"status": "ok"}

@app.post("/transcribe")
def transcribe_endpoint(
    request: Request,
    file: UploadFile = File(..., description="аудиофайл с записью разговора"),
    base_url: str | None = Form(None, description="URL OpenAI-совместимого endpoint vLLM"),
    num_speakers: int = Form(2, description="ожидаемое число спикеров"),
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
            result = transcribe(tmp_path, language="ru", base_url=base_url)
        except Exception as e:  # ошибки обращения к vLLM-серверу
            raise HTTPException(
                status_code=502,
                detail=f"Ошибка обращения к vLLM ({type(e).__name__}): {e}",
            ) from e

        try:
            # декодируем аудио один раз и переиспользуем для диаризации и эмоций
            waveform, sr = load_waveform(tmp_path)
            # пайплайн pyannote прогрет на старте (lifespan) — переиспользуем
            with _diarization_lock:
                diarization = run_diarization(
                    request.app.state.diarization_pipeline,
                    waveform,
                    sr,
                    num_speakers=num_speakers,
                )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка диаризации ({type(e).__name__}): {e}",
            ) from e


        emotions = None
        if detect_emotions:
            try:
                emotions = classify_segments(waveform, sr, result.segments)
            except Exception:
                emotions = None

        if response_format == "txt":
            return PlainTextResponse(build_text(result, diarization, emotions))
        return build_result(result, diarization, emotions)
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
