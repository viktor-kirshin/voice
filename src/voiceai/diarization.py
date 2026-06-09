from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
from huggingface_hub import get_token
from pyannote.audio import Pipeline


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str


@dataclass
class DiarizationResult:
    speakers: list[str]
    turns: list[SpeakerTurn] = field(default_factory=list)


# Модель диаризации. community-1 новее и точнее 3.1, особенно на телефонии.
# Переопределяется переменной VOICEAI_DIARIZATION_MODEL (напр. вернуть 3.1).
_DEFAULT_MODEL = "pyannote/speaker-diarization-community-1"


def load_pipeline(
    device: str = "auto",
    hf_token: str | None = None,
    model: str | None = None,
):

    token = hf_token or os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError(
            "Не найден токен HuggingFace. Выполните `hf auth login`, "
            "либо задайте переменную окружения HF_TOKEN."
        )

    model = model or os.environ.get("VOICEAI_DIARIZATION_MODEL", _DEFAULT_MODEL)
    pipeline = Pipeline.from_pretrained(model, token=token)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))
    return pipeline


def run_diarization(
    pipeline,
    waveform,
    sample_rate: int,
    num_speakers: int = 2,
) -> DiarizationResult:

    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    output = pipeline(audio_input, **kwargs)

    annotation = getattr(output, "speaker_diarization", output)

    turns = [
        SpeakerTurn(turn.start, turn.end, speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    speakers = sorted({t.speaker for t in turns})
    return DiarizationResult(speakers=speakers, turns=turns)
