# Whisper на vLLM (OpenAI-совместимый сервер)

Здесь поднимается отдельный сервис: модель **Whisper** крутится внутри
**vLLM** в Docker-контейнере и отдаёт **OpenAI-совместимый HTTP API**.
Клиент `voiceai` обращается к нему через `--backend vllm`, а диаризацию
(pyannote) и склейку делает у себя.

```
┌──────────────────────────┐         HTTP (OpenAI API)        ┌────────────────────────────┐
│  клиент voiceai (CPU/Mac)│  ── POST /v1/audio/transcriptions ─► │  Docker: vLLM + Whisper (GPU)│
│  • pyannote (диаризация) │  ◄── verbose_json: segments ────  │  • openai/whisper-large-v3   │
│  • выравнивание + вывод   │                                   │  • endpoint :8000/v1          │
└──────────────────────────┘                                   └────────────────────────────┘
```

## Требования

- Linux-хост с **NVIDIA GPU** (vLLM не работает на macOS/Apple Silicon).
- Docker + **nvidia-container-toolkit** (`--gpus all` должен работать).
- ~15–20 ГБ места под веса `whisper-large-v3` и VRAM под модель.

## Запуск сервера

```bash
# из корня проекта
HF_TOKEN=hf_xxx docker compose -f docker/docker-compose.yml up -d

# дождаться готовности и проверить
curl http://localhost:8000/v1/models
```

Первый старт долгий — vLLM скачивает веса модели в кэш HuggingFace
(том проброшен, повторно качаться не будет).

## Проверка endpoint напрямую

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@../test.mp3 \
  -F model=openai/whisper-large-v3 \
  -F response_format=verbose_json \
  -F 'timestamp_granularities[]=segment'
```

## Подключение клиента

```bash
# если сервер на этой же машине
uv run voiceai transcribe call.mp3 \
    --backend vllm \
    --model openai/whisper-large-v3

# если сервер на другом хосте
uv run voiceai transcribe call.mp3 \
    --backend vllm \
    --base-url http://GPU_HOST:8000/v1 \
    --model openai/whisper-large-v3
```

`--base-url` можно задать и переменной окружения `VOICEAI_BASE_URL`.

## Заметки

- vLLM не проверяет API-ключ; клиент шлёт `EMPTY` по умолчанию. Если перед
  vLLM стоит прокси с авторизацией — задайте `--api-key` или `OPENAI_API_KEY`.
- Для выравнивания со спикерами нужен `verbose_json` с посегментными
  таймкодами (`timestamp_granularities=[segment]`). Клиент запрашивает их сам;
  убедитесь, что версия vLLM их поддерживает.
- Сменить модель — отредактируйте `--model`/`--served-model-name` в
  `docker-compose.yml` (напр. `openai/whisper-large-v3-turbo`).
