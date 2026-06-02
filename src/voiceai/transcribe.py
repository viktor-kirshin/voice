from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Segment:
    """Один распознанный фрагмент речи."""

    start: float  # время начала, сек
    end: float    # время конца, сек
    text: str     # распознанный текст с пунктуацией


@dataclass
class Transcription:
    language: str          # определённый/заданный язык
    duration: float        # длительность аудио, сек
    segments: list[Segment]


def transcribe(
    audio_path: str,
    model: str = "openai/whisper-large-v3",
    language: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Transcription:
    """Распознаёт речь через OpenAI-совместимый endpoint vLLM.

    Whisper крутится на удалённом GPU-сервере (Docker + vLLM), сюда приходит
    только результат. POST {base_url}/audio/transcriptions — тот же контракт,
    что и у OpenAI. Запрашиваем verbose_json с посегментными таймкодами: они
    нужны, чтобы в output.py выровнять текст со спикерами от pyannote.
    """
    from openai import OpenAI

    base_url = base_url or os.environ.get("VOICEAI_BASE_URL", "http://localhost:8000/v1")
    # vLLM не проверяет ключ, но клиент OpenAI требует непустую строку.
    api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"

    client = OpenAI(base_url=base_url, api_key=api_key)

    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=model,
            file=f,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    raw_segments = getattr(resp, "segments", None) or []
    segments = [
        Segment(
            start=float(getattr(s, "start", 0.0)),
            end=float(getattr(s, "end", 0.0)),
            text=str(getattr(s, "text", "")).strip(),
        )
        for s in raw_segments
    ]

    # Если сервер вернул только текст без сегментов — кладём одной репликой.
    if not segments and getattr(resp, "text", None):
        segments = [Segment(start=0.0, end=0.0, text=resp.text.strip())]

    return Transcription(
        language=getattr(resp, "language", None) or (language or "unknown"),
        duration=float(getattr(resp, "duration", 0.0) or 0.0),
        segments=segments,
    )
