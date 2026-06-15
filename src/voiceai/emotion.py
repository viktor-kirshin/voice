from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

# Русский SER на базе HuBERT (датасет Dusha) — эмоция всей записи.
MODEL_ID = "xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned"
FEATURE_EXTRACTOR_ID = "facebook/hubert-large-ls960-ft"

NUM2EMOTION = {0: "neutral", 1: "angry", 2: "positive", 3: "sad", 4: "other"}

TARGET_SR = 16000
MAX_SECONDS = 10


class EmotionModel:
    """Обёртка над загруженной моделью эмоций (создаётся один раз в lifespan)."""

    def __init__(self, feature_extractor, model, device: str):
        self.feature_extractor = feature_extractor
        self.model = model
        self.device = device

    @torch.no_grad()
    def predict(self, filepath: str) -> tuple[str, dict[str, float]]:
        """Возвращает (эмоция записи, вероятности по всем классам)."""
        waveform, sample_rate = torchaudio.load(filepath, normalize=True)

        # стерео -> моно, иначе multi-channel ломает форму входа
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != TARGET_SR:
            waveform = torchaudio.transforms.Resample(sample_rate, TARGET_SR)(waveform)

        inputs = self.feature_extractor(
            waveform.squeeze(0).numpy(),
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
            max_length=TARGET_SR * MAX_SECONDS,
            truncation=True,
        )
        logits = self.model(inputs["input_values"].to(self.device)).logits
        probs = F.softmax(logits, dim=-1)[0]

        idx = int(torch.argmax(probs).item())
        emotion = NUM2EMOTION[idx]
        scores = {NUM2EMOTION[i]: round(float(p), 4) for i, p in enumerate(probs)}
        return emotion, scores


def load_emotion_model(device: str | None = None) -> EmotionModel:
    """Загружает feature extractor и модель. Вызывать один раз при старте сервера."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(FEATURE_EXTRACTOR_ID)
    model = HubertForSequenceClassification.from_pretrained(MODEL_ID).to(device).eval()
    return EmotionModel(feature_extractor, model, device)
