from openai import OpenAI
from elplatai.config import VLLM_BASE_URL, VLLM_API_KEY

client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY or "EMPTY")


def get_transcription_vllm(
    audio_path: str,
    language: str = "ru",
    response_format: str = "verbose_json",
    model: str = "openai/whisper-large-v3",
    timestamp_granularities: list[str] | None = None,
) -> dict:
    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model=model,
            language=language,
            response_format=response_format,
            timestamp_granularities=timestamp_granularities or ["segment"],
        )
    return {
        "text": transcription.text,
        "segments": transcription.segments,
    }
