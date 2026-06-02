from __future__ import annotations

from pathlib import Path

from .diarization import DiarizationResult, SpeakerTurn
from .profanity import find_profanity
from .transcribe import Segment, Transcription


def _ts(seconds: float) -> str:
    """Секунды → HH:MM:SS.ss"""
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{int(m):02d}:{s:05.2f}"


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _assign_speaker(seg: Segment, turns: list[SpeakerTurn]) -> str | None:
    best_speaker: str | None = None
    best_overlap = 0.0
    for turn in turns:
        ov = _overlap(seg.start, seg.end, turn.start, turn.end)
        if ov > best_overlap:
            best_overlap = ov
            best_speaker = turn.speaker
    return best_speaker


def build_text(
    transcription: Transcription,
    diarization: DiarizationResult | None = None,
) -> str:
    """Собирает читаемую расшифровку. Если передана диаризация — с спикерами."""
    turns = diarization.turns if diarization else []
    lines: list[str] = []
    for seg in transcription.segments:
        speaker = _assign_speaker(seg, turns) if turns else None
        prefix = f"{speaker}: " if speaker else ""

        profanity = find_profanity(seg.text)
        mark = f"  [мат: {', '.join(profanity)}]" if profanity else ""

        lines.append(f"[{_ts(seg.start)} → {_ts(seg.end)}] {prefix}{seg.text}{mark}")
    return "\n".join(lines)


def write_txt(
    transcription: Transcription,
    path: str | Path,
    diarization: DiarizationResult | None = None,
) -> Path:
    """Записывает расшифровку в .txt и возвращает путь к файлу."""
    path = Path(path)
    path.write_text(build_text(transcription, diarization), encoding="utf-8")
    return path
