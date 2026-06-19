from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
import torchaudio
import torch

num2emotion = {0: 'neutral', 1: 'angry', 2: 'positive', 3: 'sad', 4: 'other'}


def load_model():
    """Загружает feature extractor и модель. Вызывать один раз (в lifespan)."""
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertForSequenceClassification.from_pretrained("xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned")
    return feature_extractor, model


def predict_emotion(filepath, feature_extractor, model):
    """Эмоция всей записи (метка из num2emotion)."""
    waveform, sample_rate = torchaudio.load(filepath, normalize=True)
    if waveform.shape[0] > 1:  # стерео -> моно, иначе ломается форма входа
        waveform = waveform.mean(dim=0, keepdim=True)
    transform = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = transform(waveform)

    inputs = feature_extractor(
            waveform.squeeze(0).numpy(),
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
            max_length=16000 * 10,
            truncation=True
        )

    logits = model(inputs['input_values']).logits
    emotion = torch.argmax(logits, dim=-1)
    return num2emotion[emotion.numpy()[0]]
