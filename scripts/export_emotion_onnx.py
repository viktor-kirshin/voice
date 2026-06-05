from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Wav2Vec2FeatureExtractor,
)

MODEL_ID = os.environ.get(
    "VOICEAI_EMOTION_MODEL", "Aniemore/wavlm-emotion-russian-resd"
)
TRITON_NAME = "wavlm_emotion"
SR = 16000

ROOT = Path(__file__).resolve().parent.parent
TRITON_ROOT = ROOT / "deploy" / "triton" / "model_repository" / TRITON_NAME
VERSION_DIR = TRITON_ROOT / "1"


def _feature_extractor():
    try:
        return AutoFeatureExtractor.from_pretrained(MODEL_ID)
    except Exception:
        return Wav2Vec2FeatureExtractor(
            sampling_rate=SR, do_normalize=True, return_attention_mask=True
        )


def export_onnx() -> Path:
    print(f"[1/3] Экспорт {MODEL_ID} → ONNX (torch.onnx, может занять время)...")
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID).eval()

    class _LogitsOnly(torch.nn.Module):
        """Отдаёт только logits — чтобы ONNX-граф имел один чистый выход."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_values):
            return self.m(input_values=input_values).logits

    wrapper = _LogitsOnly(model).eval()
    dummy = torch.randn(1, SR * 3)  # 3 секунды аудио, батч 1

    VERSION_DIR.mkdir(parents=True, exist_ok=True)
    dst = VERSION_DIR / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(dst),
            input_names=["input_values"],
            output_names=["logits"],
            dynamic_axes={
                "input_values": {0: "batch", 1: "time"},
                "logits": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"      ONNX → {dst.relative_to(ROOT)}")
    return dst


def write_triton_config(onnx_path: Path) -> Path:
    type_map = {
        onnx.TensorProto.FLOAT: "TYPE_FP32",
        onnx.TensorProto.DOUBLE: "TYPE_FP64",
        onnx.TensorProto.INT64: "TYPE_INT64",
        onnx.TensorProto.INT32: "TYPE_INT32",
        onnx.TensorProto.BOOL: "TYPE_BOOL",
    }
    graph = onnx.load(str(onnx_path)).graph

    def spec(value):
        dt = type_map.get(value.type.tensor_type.elem_type, "TYPE_FP32")
        dims = [
            d.dim_value if d.dim_value > 0 else -1
            for d in value.type.tensor_type.shape.dim
        ]
        return value.name, dt, dims

    def block(items):
        rows = [
            '  {\n    name: "%s"\n    data_type: %s\n    dims: %s\n  }'
            % (name, dt, dims)
            for name, dt, dims in items
        ]
        return "[\n" + ",\n".join(rows) + "\n]"

    inputs = [spec(i) for i in graph.input]
    outputs = [spec(o) for o in graph.output]

    config = (
        f'name: "{TRITON_NAME}"\n'
        'backend: "onnxruntime"\n'
        "max_batch_size: 0\n"
        f"input {block(inputs)}\n"
        f"output {block(outputs)}\n"
    )
    path = TRITON_ROOT / "config.pbtxt"
    path.write_text(config, encoding="utf-8")
    print(f"[2/3] config.pbtxt → {path.relative_to(ROOT)}")
    print(f"      входы:  {[(n, d) for n, _, d in inputs]}")
    print(f"      выходы: {[(n, d) for n, _, d in outputs]}")
    return path


def verify_equivalence(onnx_path: Path) -> None:
    print("[3/3] Сверка PyTorch vs ONNX...")
    extractor = _feature_extractor()
    pt_model = AutoModelForAudioClassification.from_pretrained(MODEL_ID).eval()

    rng = np.random.default_rng(0)
    samples = (rng.standard_normal(SR * 3) * 0.1).astype("float32")
    inputs = extractor(samples, sampling_rate=SR, return_tensors="pt")
    input_values = inputs["input_values"]

    with torch.no_grad():
        pt_logits = pt_model(input_values=input_values).logits.numpy()

    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    ort_logits = sess.run(
        ["logits"], {"input_values": input_values.numpy()}
    )[0]

    max_diff = float(np.abs(pt_logits - ort_logits).max())
    same_argmax = bool(pt_logits.argmax(-1)[0] == ort_logits.argmax(-1)[0])
    id2label = pt_model.config.id2label
    print(f"      max|Δlogits| = {max_diff:.6e}")
    print(f"      argmax совпадает: {same_argmax} "
          f"(PyTorch={id2label[int(pt_logits.argmax(-1)[0])]}, "
          f"ONNX={id2label[int(ort_logits.argmax(-1)[0])]})")

    assert same_argmax, "argmax расходится — экспорт некорректен"
    assert max_diff < 1e-2, f"слишком большое расхождение логитов: {max_diff}"
    print("      OK: модели эквивалентны.")


def main() -> None:
    onnx_path = export_onnx()
    write_triton_config(onnx_path)
    verify_equivalence(onnx_path)
    print("\nГотово. Triton-репозиторий: "
          f"{(ROOT / 'deploy' / 'triton' / 'model_repository').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
