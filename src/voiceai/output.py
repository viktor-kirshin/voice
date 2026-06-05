from __future__ import annotations

from .diarization import DiarizationResult, SpeakerTurn
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


def build_result(
    transcription: Transcription,
    diarization: DiarizationResult | None = None,
    emotions: list[str | None] | None = None,
) -> dict:

    turns = diarization.turns if diarization else []
    segments = []
    for i, seg in enumerate(transcription.segments):
        speaker = _assign_speaker(seg, turns) if turns else None
        emotion = emotions[i] if emotions and i < len(emotions) else None
        segments.append(
            {
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "start_ts": _ts(seg.start),
                "end_ts": _ts(seg.end),
                "speaker": speaker,
                "emotion": emotion,
                "text": seg.text,
            }
        )
    return {
        "language": transcription.language,
        "duration": round(transcription.duration, 2),
        "speakers": diarization.speakers if diarization else [],
        "segments": segments,
    }


def build_text(
    transcription: Transcription,
    diarization: DiarizationResult | None = None,
    emotions: list[str | None] | None = None,
) -> str:

    lines: list[str] = []
    for s in build_result(transcription, diarization, emotions)["segments"]:
        prefix = f"{s['speaker']}: " if s["speaker"] else ""
        emo = f"  ({s['emotion']})" if s["emotion"] else ""
        lines.append(f"[{s['start_ts']} → {s['end_ts']}] {prefix}{s['text']}{emo}")
    return "\n".join(lines)
