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
    model_size: str = "small",
    language: str | None = None,
    device: str = "auto",
) -> Transcription:
  
    from faster_whisper import WhisperModel

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments_iter, info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=True,  # отсекаем тишину, чтобы модель не «выдумывала» текст
    )

    segments = [
        Segment(start=seg.start, end=seg.end, text=seg.text.strip())
        for seg in segments_iter
    ]

    return Transcription(
        language=info.language,
        duration=info.duration,
        segments=segments,
    )
