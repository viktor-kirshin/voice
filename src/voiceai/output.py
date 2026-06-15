from __future__ import annotations

from .transcribe import Transcription


def _ts(seconds: float) -> str:
    """Секунды → HH:MM:SS.ss"""
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{int(m):02d}:{s:05.2f}"


def build_result(
    transcription: Transcription,
    emotions: list[str | None] | None = None,
) -> dict:
    """Структурированный результат (для JSON): сегменты с таймкодами и эмоциями.

    `emotions` — список меток, выровненный по `transcription.segments`.
    """
    out = []
    for i, seg in enumerate(transcription.segments):
        emotion = emotions[i] if emotions and i < len(emotions) else None
        out.append(
            {
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "start_ts": _ts(seg.start),
                "end_ts": _ts(seg.end),
                "emotion": emotion,
                "text": seg.text,
            }
        )
    return {
        "language": transcription.language,
        "duration": round(transcription.duration, 2),
        "segments": out,
    }


def build_text(
    transcription: Transcription,
    emotions: list[str | None] | None = None,
) -> str:
    """Собирает читаемую расшифровку с таймкодами и эмоциями."""
    lines: list[str] = []
    for s in build_result(transcription, emotions)["segments"]:
        emo = f"  ({s['emotion']})" if s["emotion"] else ""
        lines.append(f"[{s['start_ts']} → {s['end_ts']}] {s['text']}{emo}")
    return "\n".join(lines)
