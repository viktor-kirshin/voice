from __future__ import annotations

import re
from dataclasses import dataclass

from .diarization import DiarizationResult, SpeakerTurn
from .transcribe import Transcription


@dataclass
class AlignedSegment:
    """Реплика после выравнивания: текст + его спикер."""

    start: float
    end: float
    speaker: str | None
    text: str


def _ts(seconds: float) -> str:
    """Секунды → HH:MM:SS.ss"""
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{int(m):02d}:{s:05.2f}"


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _best_speaker(start: float, end: float, turns: list[SpeakerTurn]) -> str | None:
    """Спикер, чей интервал перекрывается с [start, end] сильнее всего."""
    best_speaker: str | None = None
    best_overlap = 0.0
    for turn in turns:
        ov = _overlap(start, end, turn.start, turn.end)
        if ov > best_overlap:
            best_overlap = ov
            best_speaker = turn.speaker
    return best_speaker


def _join_words(tokens: list[str]) -> str:
    """Склеивает токены слов в текст с аккуратными пробелами/пунктуацией."""
    text = " ".join(t.strip() for t in tokens if t.strip())
    text = re.sub(r"\s+([,.!?…:;%)\]»])", r"\1", text)   # пробел перед пунктуацией
    text = re.sub(r"([(\[«])\s+", r"\1", text)            # пробел после открывающих
    return text.strip()


def align_speakers(
    transcription: Transcription,
    diarization: DiarizationResult | None = None,
) -> list[AlignedSegment]:
    """Сопоставляет текст со спикерами.

    Если есть пословные таймкоды — назначает спикера каждому слову и режет
    реплики на смене говорящего (точнее). Иначе — посегментно, по максимальному
    перекрытию.
    """
    turns = diarization.turns if diarization else []

    if transcription.words and turns:
        return _align_by_words(transcription.words, turns)

    return [
        AlignedSegment(
            seg.start,
            seg.end,
            _best_speaker(seg.start, seg.end, turns) if turns else None,
            seg.text,
        )
        for seg in transcription.segments
    ]


def _align_by_words(words, turns: list[SpeakerTurn]) -> list[AlignedSegment]:
    groups: list[dict] = []
    for w in words:
        spk = _best_speaker(w.start, w.end, turns)
        if groups and groups[-1]["speaker"] == spk:
            groups[-1]["end"] = w.end
            groups[-1]["tokens"].append(w.text)
        else:
            groups.append(
                {"speaker": spk, "start": w.start, "end": w.end, "tokens": [w.text]}
            )
    return [
        AlignedSegment(g["start"], g["end"], g["speaker"], _join_words(g["tokens"]))
        for g in groups
    ]


def build_result(
    transcription: Transcription,
    segments: list[AlignedSegment],
    emotions: list[str | None] | None = None,
) -> dict:
    """Структурированный результат (для JSON): реплики со спикерами и эмоциями.

    `segments` — выровненные реплики (align_speakers), `emotions` — метки,
    выровненные по ним.
    """
    out = []
    for i, s in enumerate(segments):
        emotion = emotions[i] if emotions and i < len(emotions) else None
        out.append(
            {
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "start_ts": _ts(s.start),
                "end_ts": _ts(s.end),
                "speaker": s.speaker,
                "emotion": emotion,
                "text": s.text,
            }
        )
    speakers = sorted({s.speaker for s in segments if s.speaker})
    return {
        "language": transcription.language,
        "duration": round(transcription.duration, 2),
        "speakers": speakers,
        "segments": out,
    }


def build_text(
    transcription: Transcription,
    segments: list[AlignedSegment],
    emotions: list[str | None] | None = None,
) -> str:
    """Собирает читаемую расшифровку с таймкодами, спикерами и эмоциями."""
    lines: list[str] = []
    for s in build_result(transcription, segments, emotions)["segments"]:
        prefix = f"{s['speaker']}: " if s["speaker"] else ""
        emo = f"  ({s['emotion']})" if s["emotion"] else ""
        lines.append(f"[{s['start_ts']} → {s['end_ts']}] {prefix}{s['text']}{emo}")
    return "\n".join(lines)
