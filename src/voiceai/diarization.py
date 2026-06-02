from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field

_TARGET_SR = 16000


def _load_waveform(audio_path: str):

    import numpy as np
    import torch

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "Не найден ffmpeg. Установите его: `brew install ffmpeg`."
        )

    cmd = [
        ffmpeg, "-nostdin", "-threads", "0",
        "-i", audio_path,
        "-f", "f32le",            
        "-acodec", "pcm_f32le",
        "-ac", "1",               
        "-ar", str(_TARGET_SR), 
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "ignore").strip().splitlines()[-1:] or [""]
        raise RuntimeError(f"ffmpeg не смог декодировать {audio_path}: {msg[0]}")

    samples = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    waveform = torch.from_numpy(samples).unsqueeze(0)  # (1, N)
    return waveform, _TARGET_SR


@dataclass
class SpeakerTurn:

    start: float
    end: float
    speaker: str


@dataclass
class DiarizationResult:
    speakers: list[str]
    turns: list[SpeakerTurn] = field(default_factory=list)


def diarize(
    audio_path: str,
    num_speakers: int = 2,
    device: str = "auto",
    hf_token: str | None = None,
) -> DiarizationResult:

    import torch
    from huggingface_hub import get_token
    from pyannote.audio import Pipeline


    token = hf_token or os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError(
            "Не найден токен HuggingFace. Выполните `hf auth login`, "
            "либо задайте переменную окружения HF_TOKEN."
        )

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=token
    )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))

    waveform, sample_rate = _load_waveform(audio_path)
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
