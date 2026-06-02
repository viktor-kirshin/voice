# Whisper на vLLM (OpenAI-совместимый сервер)

Здесь поднимается отдельный сервис: модель **Whisper** крутится внутри
**vLLM** в Docker-контейнере и отдаёт **OpenAI-совместимый HTTP API**.
Сервис `voiceai` обращается к нему по сети (поле `base_url`), а диаризацию
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

## Подключение из сервиса voiceai

FastAPI-сервис `voiceai` обращается к этому endpoint при обработке `/transcribe`.
Адрес vLLM передаётся в поле `base_url` (или переменной `VOICEAI_BASE_URL`):

```bash
# сервер vLLM на этой же машине (base_url по умолчанию http://localhost:8000/v1)
curl -X POST http://localhost:8080/transcribe -F file=@../test.mp3

# vLLM на другом хосте
curl -X POST http://localhost:8080/transcribe \
    -F file=@../test.mp3 \
    -F base_url=http://GPU_HOST:8000/v1
```

## Заметки

- vLLM не проверяет API-ключ; сервис шлёт `EMPTY` по умолчанию. Если перед
  vLLM стоит прокси с авторизацией — задайте поле `api_key` или `OPENAI_API_KEY`.
- Для выравнивания со спикерами нужен `verbose_json` с посегментными
  таймкодами (`timestamp_granularities=[segment]`). Клиент запрашивает их сам;
  убедитесь, что версия vLLM их поддерживает.
- Сменить модель — отредактируйте `--model`/`--served-model-name` в
  `docker-compose.yml` (напр. `openai/whisper-large-v3-turbo`).
