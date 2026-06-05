from __future__ import annotations

import shutil
import subprocess

import numpy as np
import torch

TARGET_SR = 16000


def load_waveform(audio_path: str):

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "Не найден ffmpeg. Установите его: `brew install ffmpeg`."
        )

    cmd = [
        ffmpeg, "-nostdin", "-threads", "0",
        "-i", audio_path,
        "-f", "f32le",            # сырой float32 little-endian на stdout
        "-acodec", "pcm_f32le",
        "-ac", "1",               # моно (downmix)
        "-ar", str(TARGET_SR),    # ресемпл в 16 кГц
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "ignore").strip().splitlines()[-1:] or [""]
        raise RuntimeError(f"ffmpeg не смог декодировать {audio_path}: {msg[0]}")

    samples = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    waveform = torch.from_numpy(samples).unsqueeze(0)  # (1, N)
    return waveform, TARGET_SR
