from __future__ import annotations

import os
from dataclasses import dataclass, field

from openai import OpenAI


@dataclass
class Segment:
    """Один распознанный фрагмент речи."""

    start: float  # время начала, сек
    end: float    # время конца, сек
    text: str     # распознанный текст с пунктуацией


@dataclass
class Word:
    """Одно слово с таймкодами (для выравнивания со спикерами по словам)."""

    start: float
    end: float
    text: str


@dataclass
class Transcription:
    language: str               # определённый/заданный язык
    duration: float             # длительность аудио, сек
    segments: list[Segment]
    words: list[Word] = field(default_factory=list)  # пусто, если сервер не отдал


def _create(client, model, file_tuple, language, temperature, granularities, prompt):
    kwargs = dict(
        model=model,
        file=file_tuple,
        language=language,
        temperature=temperature,
        response_format="verbose_json",
        timestamp_granularities=granularities,
    )
    if prompt:
        kwargs["prompt"] = prompt
    return client.audio.transcriptions.create(**kwargs)


def transcribe(
    audio_path: str,
    model: str = "openai/whisper-large-v3",
    language: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    prompt: str | None = None,
    temperature: float = 0.0,
) -> Transcription:
    """Распознаёт речь через OpenAI-совместимый endpoint vLLM.

    prompt — подсказка Whisper (имена, термины) для меньшей путаницы слов.
    Запрашивает пословные таймкоды; если сервер их не поддерживает —
    откатывается на посегментные.
    """
    base_url = base_url or os.environ.get("VOICEAI_BASE_URL", "http://localhost:8000/v1")
    # vLLM не проверяет ключ, но клиент OpenAI требует непустую строку.
    api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    prompt = prompt if prompt is not None else os.environ.get("VOICEAI_PROMPT")

    client = OpenAI(base_url=base_url, api_key=api_key)

    with open(audio_path, "rb") as f:
        file_tuple = (os.path.basename(audio_path), f.read())

    # Пробуем с пословными таймкодами; если vLLM их не умеет — посегментно.
    try:
        resp = _create(client, model, file_tuple, language, temperature,
                       ["segment", "word"], prompt)
    except Exception:
        resp = _create(client, model, file_tuple, language, temperature,
                       ["segment"], prompt)

    raw_segments = getattr(resp, "segments", None) or []
    segments = [
        Segment(
            start=float(getattr(s, "start", 0.0)),
            end=float(getattr(s, "end", 0.0)),
            text=str(getattr(s, "text", "")).strip(),
        )
        for s in raw_segments
    ]

    raw_words = getattr(resp, "words", None) or []
    words = [
        Word(
            start=float(getattr(w, "start", 0.0)),
            end=float(getattr(w, "end", 0.0)),
            text=str(getattr(w, "word", getattr(w, "text", ""))),
        )
        for w in raw_words
    ]

    # Если сервер вернул только текст без сегментов — кладём одной репликой.
    if not segments and getattr(resp, "text", None):
        segments = [Segment(start=0.0, end=0.0, text=resp.text.strip())]

    return Transcription(
        language=getattr(resp, "language", None) or (language or "unknown"),
        duration=float(getattr(resp, "duration", 0.0) or 0.0),
        segments=segments,
        words=words,
    )
