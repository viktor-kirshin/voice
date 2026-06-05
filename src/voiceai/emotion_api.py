from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse

from .emotion import classify_segments
from .transcribe import Segment

app = FastAPI(
    title="voiceai-emotion",
    description=(
        "Распознавание эмоций речи на русском (Aniemore/WavLM, датасет RESD). "
        "Принимает аудио и (опционально) список сегментов, возвращает эмоции."
    ),
    version="0.1.0",
)

# Признак «весь файл» для сегмента без явного конца.
_WHOLE_FILE_END = 1e9


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Корень → Swagger UI."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    """Проверка живости сервиса."""
    return {"status": "ok"}


@app.post("/emotion")
def emotion_endpoint(
    file: UploadFile = File(..., description="аудиофайл"),
    segments: str | None = Form(
        None,
        description='JSON-список сегментов: [{"start":0.0,"end":4.0}, ...]. '
        "Если пусто — классифицируется весь файл (до 30 c).",
    ),
    device: str = Form("auto", description="устройство: auto/cpu/cuda"),
):
    """Определяет эмоцию по аудио.

    - С `segments` — возвращает метку на каждый сегмент.
    - Без `segments` — одну метку на весь файл.
    """
    suffix = Path(file.filename or "audio").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        if segments:
            try:
                raw = json.loads(segments)
                segs = [
                    Segment(float(s["start"]), float(s["end"]), "") for s in raw
                ]
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Некорректный JSON в segments: {e}",
                ) from e
        else:
            segs = [Segment(0.0, _WHOLE_FILE_END, "")]

        try:
            labels = classify_segments(tmp_path, segs, device=device)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка классификации ({type(e).__name__}): {e}",
            ) from e

        if not segments:
            return {"emotion": labels[0]}

        return {
            "segments": [
                {"start": s.start, "end": s.end, "emotion": label}
                for s, label in zip(segs, labels)
            ]
        }
    finally:
        os.unlink(tmp_path)


def run() -> None:
    """Точка входа `voiceai-emotion`: поднимает HTTP-сервер uvicorn."""
    import uvicorn

    uvicorn.run(
        "voiceai.emotion_api:app",
        host=os.environ.get("VOICEAI_EMOTION_HOST", "0.0.0.0"),
        port=int(os.environ.get("VOICEAI_EMOTION_PORT", "8081")),
    )


if __name__ == "__main__":
    run()
