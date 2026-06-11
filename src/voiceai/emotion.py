import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, AutoModel, Wav2Vec2FeatureExtractor

MODEL_PATH = "Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition"
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def speech_file_to_array(path, sampling_rate=16000):
    speech_array, sr = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sr, sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech


def predict(path, sampling_rate=16000):
    speech = speech_file_to_array(path, sampling_rate)
    inputs = feature_extractor(speech, sampling_rate=sampling_rate,
                               return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    results = [
        {"Emotion": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"}
        for i, score in enumerate(scores)
    ]
    return results

result = predict("1.waw", 16000)
print(result)