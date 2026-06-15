from __future__ import annotations

from .transcribe import Transcription


def _ts(seconds: float) -> str:
    """Секунды → HH:MM:SS.ss"""
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{int(m):02d}:{s:05.2f}"


def build_result(
    transcription: Transcription,
    emotion: str | None = None,
) -> dict:
    """Единый JSON: язык, длительность, эмоция всей записи и сегменты текста."""
    segments = [
        {
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "start_ts": _ts(seg.start),
            "end_ts": _ts(seg.end),
            "text": seg.text,
        }
        for seg in transcription.segments
    ]
    return {
        "language": transcription.language,
        "duration": round(transcription.duration, 2),
        "emotion": emotion,
        "segments": segments,
    }
