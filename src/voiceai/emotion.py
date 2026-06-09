from __future__ import annotations
import os
import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Wav2Vec2FeatureExtractor,
)

from .transcribe import Segment

_DEFAULT_MODEL = "Aniemore/wavlm-emotion-russian-resd"
_SR = 16000
_MIN_SECONDS = 0.3
_MAX_SECONDS = 30.0

_cache: dict = {}


def _get_model(model_name: str, device: str):
    key = (model_name, device)
    if key not in _cache:
        try:
            extractor = AutoFeatureExtractor.from_pretrained(model_name)
        except Exception:
            extractor = Wav2Vec2FeatureExtractor(
                sampling_rate=_SR, do_normalize=True, return_attention_mask=True
            )

        model = AutoModelForAudioClassification.from_pretrained(model_name)
        model.to(device)
        model.eval()
        _cache[key] = (model, extractor)
    return _cache[key]

def classify_segments(
    waveform,
    sample_rate: int,
    segments: list[Segment],
    model_name: str | None = None,
    device: str | None = None,
) -> list[str | None]:

    if not segments:
        return []

    model_name = model_name or os.environ.get("VOICEAI_EMOTION_MODEL", _DEFAULT_MODEL)
    device = device or os.environ.get("VOICEAI_DEVICE", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, extractor = _get_model(model_name, device)

    samples = waveform[0]
    total = samples.shape[0]
    min_len = int(_MIN_SECONDS * sample_rate)
    max_len = int(_MAX_SECONDS * sample_rate)

    labels: list[str | None] = []
    for seg in segments:
        a = max(0, int(seg.start * sample_rate))
        b = min(total, int(seg.end * sample_rate))
        chunk = samples[a : min(b, a + max_len)]
        if chunk.shape[0] < min_len:
            labels.append(None)
            continue

        inputs = extractor(chunk.numpy(), sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        idx = int(logits.softmax(dim=-1).argmax(dim=-1).item())
        labels.append(model.config.id2label[idx])

    return labels
