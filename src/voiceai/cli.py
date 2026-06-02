import argparse
import sys
from pathlib import Path

from .transcribe import transcribe
from .diarization import diarize
from .output import write_txt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voiceai")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcribe", help="расшифровать аудиофайл в текст")
    t.add_argument("audio", help="путь к аудиофайлу")
    t.add_argument("--model", default="openai/whisper-large-v3",
                   help="id модели Whisper на vLLM-сервере")
    t.add_argument("--language", default=None, help="код языка (ru/en/...) или auto")
    t.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="устройство для диаризации pyannote")
    t.add_argument("--output", default=None, help="путь к выходному .txt")
    t.add_argument("--base-url", default=None,
                   help="URL OpenAI-совместимого endpoint vLLM, "
                        "напр. http://localhost:8000/v1")
    t.add_argument("--api-key", default=None,
                   help="ключ для endpoint (vLLM игнорирует; по умолчанию EMPTY)")

    args = parser.parse_args(argv)

    audio = Path(args.audio)
    if not audio.exists():
        print(f"Файл не найден: {audio}", file=sys.stderr)
        return 1

    language = None if args.language in (None, "auto") else args.language

    result = transcribe(
        str(audio),
        model=args.model,
        language=language,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    diarization = diarize(str(audio), device=args.device)

    out_path = Path(args.output) if args.output else audio.with_suffix(".txt")
    write_txt(result, out_path, diarization)
    print(f"Готово. Язык: {result.language}. Текст → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
