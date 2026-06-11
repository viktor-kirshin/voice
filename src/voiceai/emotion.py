import torch
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2FeatureExtractor
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from huggingface_hub import snapshot_download

MODEL_PATH = "Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition"

local_path = snapshot_download(MODEL_PATH)

config_class = get_class_from_dynamic_module(
    "wav2vec2fsr_config.W2V2FSRConfig",
    local_path
)
model_class = get_class_from_dynamic_module(
    "wav2vec2speechclassification.Wav2Vec2ForSpeechClassification",
    local_path
)

config = config_class.from_pretrained(local_path)
model = model_class.from_pretrained(local_path, config=config)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(local_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def speech_file_to_array(path, sampling_rate=16000):
    speech_array, sr = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sr, sampling_rate)
    return resampler(speech_array).squeeze().numpy()


def predict(path, sampling_rate=16000):
    speech = speech_file_to_array(path, sampling_rate)
    inputs = feature_extractor(
        speech,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    results = [
        {"emotion": config.id2label[i], "score": f"{round(score * 100, 1)}%"}
        for i, score in enumerate(scores)
    ]
    return sorted(results, key=lambda x: float(x["score"][:-1]), reverse=True)

result = predict("1.waw", 16000)
print(result)