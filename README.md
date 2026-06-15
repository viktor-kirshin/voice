# voiceai — транскрибация диалогов с распознаванием эмоций

**FastAPI-сервис**: принимает аудиозапись разговора, возвращает JSON с
расшифровкой и эмоцией записи. Под капотом:

- **Whisper** на сервере **vLLM** (OpenAI-совместимый endpoint) — речь в текст
  с пунктуацией и таймкодами;
- модель **эмоций** (HuBERT, дообученная на русском датасете Dusha) — эмоция
  всей записи.

Язык фиксирован — **русский**, формат ответа — всегда **JSON**.

## Как это работает

```
                 ┌─────────────── FastAPI voiceai ───────────────┐
аудиофайл ──POST /transcribe──►                                   │
                 │  Whisper @ vLLM  ── HTTP ─► GPU-сервер (vLLM)   │
                 │     «что сказано» → сегменты [start, end, text] │
                 │                                                 │
                 │  HuBERT (Dusha)  ── локально                    │
                 │     «эмоция записи» → метка + вероятности        │
                 │                                                 │
                 └────────► объединённый JSON ◄────────────────────┘
```

- **Распознавание речи** вынесено на vLLM: тяжёлая модель Whisper крутится на
  GPU-сервере как сервис с OpenAI-контрактом, сам `voiceai` остаётся лёгким.
- **Эмоция** считается локально по всему файлу моделью
  `xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned`
  (классы: `neutral`, `angry`, `positive`, `sad`, `other`). Модель грузится
  один раз при старте сервера (lifespan) и переиспользуется.

## Стек

- **Python 3.13**
- **uv** — окружение и зависимости
- `fastapi` + `uvicorn` — HTTP-сервис и ASGI-сервер
- `python-multipart` — приём загружаемых файлов
- `openai` — клиент к OpenAI-совместимому endpoint Whisper (vLLM)
- `transformers` + `torch` + `torchaudio` — модель эмоций (HuBERT) и чтение аудио
- **vLLM в Docker** — сервис, на котором крутится Whisper

## Установка

```bash
uv sync
# для чтения mp3 моделью эмоций может понадобиться ffmpeg:
brew install ffmpeg        # macOS (Linux: apt-get install -y ffmpeg)
```

## Запуск

1. Поднять Whisper в vLLM (GPU-машина, см. [docker/README.md](docker/README.md)):

```bash
docker compose -f docker/docker-compose.yml up -d
curl http://localhost:8000/v1/models     # проверка
```

2. Указать сервису адрес vLLM и запустить:

```bash
export VOICEAI_BASE_URL=http://GPU_HOST:8000/v1   # или http://localhost:8000/v1
uv run voiceai                                     # сервис на :8080
```

Документация (Swagger UI) — `http://localhost:8080/docs`. Хост/порт меняются
через `VOICEAI_HOST` / `VOICEAI_PORT`.

## API

| Метод | Путь | Назначение |
|---|---|---|
| `GET` | `/health` | проверка живости |
| `POST` | `/transcribe` | загрузить аудио → JSON с текстом и эмоцией |
| `GET` | `/docs` | Swagger UI |

### `POST /transcribe`

`multipart/form-data`:

| Поле | Описание | По умолчанию |
|---|---|---|
| `file` | аудиофайл с записью разговора (обязательно) | — |
| `base_url` | URL OpenAI-совместимого endpoint vLLM | `http://localhost:8000/v1` |
| `prompt` | подсказка Whisper (имена, термины) для точности | — |
| `detect_emotions` | определять эмоцию записи | `true` |

`base_url` можно задать и переменной `VOICEAI_BASE_URL`.

Пример:

```bash
curl -X POST http://localhost:8080/transcribe -F file=@dialog.wav
```

## Формат ответа (JSON)

```json
{
  "language": "ru",
  "duration": 12.3,
  "emotion": "neutral",
  "emotion_scores": {
    "neutral": 0.81, "angry": 0.05, "positive": 0.08, "sad": 0.04, "other": 0.02
  },
  "segments": [
    {"start": 0.0, "end": 4.85, "start_ts": "00:00:00.00",
     "end_ts": "00:00:04.85", "text": "Привет, как у тебя дела?"}
  ]
}
```

## Структура проекта

```
dir1/
├── README.md
├── pyproject.toml
├── uv.lock
├── docker/                  # vLLM + Whisper (OpenAI-совместимый endpoint)
│   ├── docker-compose.yml
│   └── README.md
└── src/voiceai/
    ├── api.py          # FastAPI: /health, /transcribe, точка входа voiceai
    ├── transcribe.py   # Whisper через OpenAI-совместимый endpoint vLLM
    ├── emotion.py      # эмоция записи (HuBERT, датасет Dusha)
    └── output.py       # сборка итогового JSON
```

## Заметки

- **Whisper выполняется только на vLLM** (Linux + NVIDIA GPU). Без доступного
  vLLM `/transcribe` вернёт `502`.
- Модель эмоций грузится **при старте сервера** (lifespan, веса скачиваются
  один раз) и переиспользуется всеми запросами. Если веса не скачались —
  сервис не поднимется (fail-fast).
- Эмоция определяется **по всей записи** (модель файл-уровневая, анализирует
  первые ~10 секунд).
- Аудио для модели эмоций читается через **torchaudio** (приводится к 16 кГц
  моно).

## Дальнейшие шаги

- Эмоции по сегментам/спикерам вместо одной на запись.
- Разделение спикеров (диаризация) при необходимости.
- Фоновая обработка длинных записей (очередь задач).
