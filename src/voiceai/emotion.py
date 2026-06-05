from __future__ import annotations

import os

from .diarization import _load_waveform
from .transcribe import Segment

_DEFAULT_MODEL = "Aniemore/wavlm-emotion-russian-resd"
_SR = 16000
_MIN_SECONDS = 0.3   # слишком короткие куски не классифицируем
_MAX_SECONDS = 30.0  # ограничиваем длину одного куска, чтобы не словить OOM

# Кэш загруженных моделей: (model_name, device) -> (model, feature_extractor)
_cache: dict = {}


def _get_model(model_name: str, device: str):
    key = (model_name, device)
    if key not in _cache:
        import torch
        from transformers import (
            AutoFeatureExtractor,
            AutoModelForAudioClassification,
        )

        try:
            extractor = AutoFeatureExtractor.from_pretrained(model_name)
        except Exception:
            # некоторые репозитории не содержат preprocessor_config — берём
            # стандартный экстрактор wav2vec2/wavlm (16 кГц).
            from transformers import Wav2Vec2FeatureExtractor

            extractor = Wav2Vec2FeatureExtractor(
                sampling_rate=_SR, do_normalize=True, return_attention_mask=True
            )

        model = AutoModelForAudioClassification.from_pretrained(model_name)
        model.to(device)
        model.eval()
        _cache[key] = (model, extractor)
    return _cache[key]


def classify_segments(
    audio_path: str,
    segments: list[Segment],
    model_name: str | None = None,
    device: str = "auto",
) -> list[str | None]:

    if not segments:
        return []

    import torch

    model_name = model_name or os.environ.get("VOICEAI_EMOTION_MODEL", _DEFAULT_MODEL)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, extractor = _get_model(model_name, device)

    waveform, sr = _load_waveform(audio_path)  # (1, N) float32 @ 16 кГц
    samples = waveform[0]
    total = samples.shape[0]
    min_len = int(_MIN_SECONDS * sr)
    max_len = int(_MAX_SECONDS * sr)

    labels: list[str | None] = []
    for seg in segments:
        a = max(0, int(seg.start * sr))
        b = min(total, int(seg.end * sr))
        chunk = samples[a : min(b, a + max_len)]
        if chunk.shape[0] < min_len:
            labels.append(None)
            continue

        inputs = extractor(
            chunk.numpy(), sampling_rate=sr, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        idx = int(logits.softmax(dim=-1).argmax(dim=-1).item())
        labels.append(model.config.id2label[idx])

    return labels
