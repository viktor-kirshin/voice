from typing import Literal
from pydantic import BaseModel, Field
from elplatai.clients.vllm_client import get_transcription_vllm


class AskWhisperAction(BaseModel):
    action: Literal["ask_whisper"]
    audio_path: str = Field(min_length=1)
    language: str = "ru"
    response_format: Literal["json", "verbose_json", "text", "srt", "vtt"] = "verbose_json"
    model: str = "openai/whisper-large-v3"
    timestamp_granularities: list[Literal["segment", "word"]] | None = None


def ask_whisper(request: AskWhisperAction) -> dict:
    kwargs = {
        "audio_path": request.audio_path,
        "language": request.language,
        "response_format": request.response_format,
        "model": request.model,
        "timestamp_granularities": request.timestamp_granularities,
    }
    return get_transcription_vllm(**kwargs)
