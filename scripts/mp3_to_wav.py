"""Конвертация всех mp3-файлов в wav через ffmpeg.

По умолчанию ищет .mp3 в текущей папке и кладёт .wav рядом, в формате
16 кГц / моно (как использует пайплайн voiceai). Параметры настраиваются.

Примеры:
    # все mp3 в текущей папке → wav рядом (16 кГц, моно)
    uv run python scripts/mp3_to_wav.py

    # конкретные файлы/папки, с рекурсией и в отдельную папку
    uv run python scripts/mp3_to_wav.py . --recursive --outdir wav

    # сохранить исходную частоту и стерео
    uv run python scripts/mp3_to_wav.py audio/ --sample-rate 0 --channels 0
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def find_mp3(paths: list[str], recursive: bool) -> list[Path]:
    """Собирает .mp3 из переданных файлов и папок (без дубликатов)."""
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.suffix.lower() == ".mp3":
            found.append(p)
        elif p.is_dir():
            pattern = "**/*.mp3" if recursive else "*.mp3"
            found.extend(sorted(p.glob(pattern)))
        else:
            print(f"Пропуск (не найдено или не mp3): {p}", file=sys.stderr)
    # убираем дубликаты, сохраняя порядок
    return list(dict.fromkeys(found))


def convert_one(
    ffmpeg: str,
    src: Path,
    dst: Path,
    sample_rate: int,
    channels: int,
    overwrite: bool,
) -> bool:
    """Конвертирует один файл. Возвращает True при успехе."""
    if dst.exists() and not overwrite:
        print(f"= пропуск (уже есть): {dst}")
        return True

    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-nostdin", "-y" if overwrite else "-n", "-i", str(src)]
    if sample_rate > 0:
        cmd += ["-ar", str(sample_rate)]
    if channels > 0:
        cmd += ["-ac", str(channels)]
    cmd += ["-c:a", "pcm_s16le", str(dst)]  # стандартный 16-бит PCM wav

    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "ignore").strip().splitlines()[-1:] or [""]
        print(f"✗ ошибка: {src} → {tail[0]}", file=sys.stderr)
        return False

    print(f"✓ {src}  →  {dst}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Конвертация mp3 → wav (ffmpeg)")
    parser.add_argument("paths", nargs="*", default=["."],
                        help="файлы или папки с mp3 (по умолчанию текущая папка)")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="искать mp3 в подпапках")
    parser.add_argument("--outdir", default=None,
                        help="папка для wav (по умолчанию — рядом с исходником)")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="частота wav, Гц; 0 — оставить исходную (по умолч. 16000)")
    parser.add_argument("--channels", type=int, default=1,
                        help="число каналов; 0 — оставить как есть (по умолч. 1 = моно)")
    parser.add_argument("--overwrite", action="store_true",
                        help="перезаписывать существующие wav")
    args = parser.parse_args(argv)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("Не найден ffmpeg. Установите его: `brew install ffmpeg`.",
              file=sys.stderr)
        return 1

    files = find_mp3(args.paths, args.recursive)
    if not files:
        print("Файлы .mp3 не найдены.", file=sys.stderr)
        return 1

    outdir = Path(args.outdir) if args.outdir else None
    print(f"Найдено mp3: {len(files)}")

    ok = 0
    for src in files:
        dst = (outdir / f"{src.stem}.wav") if outdir else src.with_suffix(".wav")
        if convert_one(ffmpeg, src, dst, args.sample_rate, args.channels,
                       args.overwrite):
            ok += 1

    print(f"\nГотово: {ok}/{len(files)} сконвертировано.")
    return 0 if ok == len(files) else 2


if __name__ == "__main__":
    raise SystemExit(main())
