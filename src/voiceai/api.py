from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse

from .emotion import load_model, predict_emotion
from .output import build_result
from .transcribe import transcribe

logger = logging.getLogger("voiceai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Прогрев модели эмоций один раз при старте сервера.

    feature extractor и модель грузятся здесь и переиспользуются всеми
    запросами (в app.state). Если веса не скачались — сервис падает сразу
    на старте (fail-fast), а не отдаёт 500 в середине обработки.
    """
    app.state.emotion = load_model()  # (feature_extractor, model)
    yield
    app.state.emotion = None


app = FastAPI(
    title="voiceai",
    version="0.1.0",
    lifespan=lifespan,
)


INDEX_HTML = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>voiceai — транскрибация</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --fg:#e2e8f0; --muted:#94a3b8;
          --accent:#38bdf8; --border:#334155; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         background:var(--bg); color:var(--fg); line-height:1.5; }
  .wrap { max-width: 820px; margin: 0 auto; padding: 32px 16px 64px; }
  h1 { font-size: 1.5rem; margin: 0 0 4px; }
  .sub { color: var(--muted); margin: 0 0 24px; }
  .sub a { color: var(--accent); }
  .card { background:var(--card); border:1px solid var(--border); border-radius:12px;
          padding:20px; }
  label { display:block; font-size:.85rem; color:var(--muted); margin:14px 0 6px; }
  input[type=text], input[type=file] { width:100%; padding:10px 12px;
          background:#0b1220; color:var(--fg); border:1px solid var(--border);
          border-radius:8px; font-size:.95rem; }
  .row { display:flex; align-items:center; gap:8px; margin-top:14px; }
  .row label { margin:0; color:var(--fg); }
  button { margin-top:20px; width:100%; padding:12px; font-size:1rem; font-weight:600;
           background:var(--accent); color:#04293a; border:0; border-radius:8px;
           cursor:pointer; }
  button:disabled { opacity:.6; cursor:default; }
  .result { margin-top:24px; display:none; }
  .badge { display:inline-block; padding:4px 12px; border-radius:999px;
           background:#0b1220; border:1px solid var(--border); font-weight:600; }
  .meta { color:var(--muted); font-size:.9rem; margin:8px 0 16px; }
  .seg { padding:10px 12px; border:1px solid var(--border); border-radius:8px;
         margin-bottom:8px; background:#0b1220; }
  .seg .ts { color:var(--accent); font-size:.8rem; font-family:ui-monospace, monospace; }
  pre { background:#0b1220; border:1px solid var(--border); border-radius:8px;
        padding:12px; overflow:auto; font-size:.8rem; }
  details { margin-top:16px; }
  summary { cursor:pointer; color:var(--muted); }
  .err { color:#f87171; white-space:pre-wrap; }
</style>
</head>
<body>
<div class="wrap">
  <h1>voiceai — транскрибация диалога</h1>
  <p class="sub">Загрузите аудио и укажите адрес vLLM. <a href="/docs">Swagger UI →</a></p>

  <div class="card">
    <form id="form">
      <label for="file">Аудиофайл (wav / mp3 …)</label>
      <input id="file" type="file" accept="audio/*" required>

      <label for="base_url">base_url (OpenAI-совместимый endpoint vLLM)</label>
      <input id="base_url" type="text" placeholder="http://localhost:8000/v1">

      <label for="prompt">prompt (необязательно — имена, термины)</label>
      <input id="prompt" type="text" placeholder="">

      <div class="row">
        <input id="detect_emotions" type="checkbox" checked>
        <label for="detect_emotions">определять эмоцию записи</label>
      </div>

      <button id="submit" type="submit">Распознать</button>
    </form>
  </div>

  <div class="result card" id="result"></div>
</div>

<script>
const form = document.getElementById('form');
const btn = document.getElementById('submit');
const result = document.getElementById('result');

function esc(s) {
  return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = document.getElementById('file').files[0];
  if (!file) return;

  const fd = new FormData();
  fd.append('file', file);
  const baseUrl = document.getElementById('base_url').value.trim();
  if (baseUrl) fd.append('base_url', baseUrl);
  const prompt = document.getElementById('prompt').value.trim();
  if (prompt) fd.append('prompt', prompt);
  fd.append('detect_emotions', document.getElementById('detect_emotions').checked);

  btn.disabled = true;
  btn.textContent = 'Обработка…';
  result.style.display = 'block';
  result.innerHTML = '<span class="meta">Распознаём, это может занять время…</span>';

  try {
    const res = await fetch('/transcribe', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) {
      result.innerHTML = '<div class="err">Ошибка ' + res.status + ': '
        + esc(data.detail || JSON.stringify(data)) + '</div>';
      return;
    }
    render(data);
  } catch (err) {
    result.innerHTML = '<div class="err">Сбой запроса: ' + esc(err.message) + '</div>';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Распознать';
  }
});

function render(data) {
  let html = '';
  if (data.emotion) html += '<span class="badge">эмоция: ' + esc(data.emotion) + '</span>';
  html += '<div class="meta">язык: ' + esc(data.language)
        + ' · длительность: ' + esc(data.duration) + ' c · сегментов: '
        + (data.segments ? data.segments.length : 0) + '</div>';

  for (const s of (data.segments || [])) {
    html += '<div class="seg"><div class="ts">' + esc(s.start_ts) + ' → '
          + esc(s.end_ts) + '</div>' + esc(s.text) + '</div>';
  }

  html += '<details><summary>Показать JSON</summary><pre>'
        + esc(JSON.stringify(data, null, 2)) + '</pre></details>';
  result.innerHTML = html;
}
</script>
</body>
</html>"""


@app.get("/", include_in_schema=False)
def root() -> HTMLResponse:
    """Простая страница для загрузки аудио и просмотра результата."""
    return HTMLResponse(INDEX_HTML)


@app.get("/health")
def health() -> dict:
    """Проверка живости сервиса."""
    return {"status": "ok"}


@app.post("/transcribe")
def transcribe_endpoint(
    request: Request,
    file: UploadFile = File(..., description="аудиофайл с записью разговора"),
    base_url: str | None = Form(None, description="URL OpenAI-совместимого endpoint vLLM"),
    prompt: str | None = Form(None, description="подсказка Whisper: имена/термины для точности"),
    detect_emotions: bool = Form(True, description="определять эмоцию записи"),
):
    """Распознаёт речь (Whisper@vLLM) и эмоцию записи. Ответ — JSON, язык русский."""
    suffix = Path(file.filename or "audio").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        try:
            result = transcribe(tmp_path, language="ru", base_url=base_url, prompt=prompt)
        except Exception as e:  # ошибки обращения к vLLM-серверу
            raise HTTPException(
                status_code=502,
                detail=f"Ошибка обращения к vLLM ({type(e).__name__}): {e}",
            ) from e

        # эмоция необязательна: при сбое не валим запрос, просто без неё
        emotion = None
        if detect_emotions:
            try:
                feature_extractor, model = request.app.state.emotion
                emotion = predict_emotion(tmp_path, feature_extractor, model)
            except Exception:
                logger.exception("Не удалось определить эмоцию")
                emotion = None

        return build_result(result, emotion)
    finally:
        os.unlink(tmp_path)


def run() -> None:
    uvicorn.run(
        "voiceai.api:app",
        host=os.environ.get("VOICEAI_HOST", "0.0.0.0"),
        port=int(os.environ.get("VOICEAI_PORT", "8080")),
    )


if __name__ == "__main__":
    run()
