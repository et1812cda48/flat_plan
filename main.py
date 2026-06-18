import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

app = FastAPI(
    title="Floor Plan Wall Extractor",
    description="Upload a floor plan image and receive the coordinates of detected walls.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HTML_PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>ПланАнализ — Экстрактор стен</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

  :root {
    --orange: #F5820A;
    --orange-dark: #C96800;
    --concrete: #2B2B2B;
    --concrete-mid: #3A3A3A;
    --concrete-light: #4E4E4E;
    --steel: #8A9099;
    --steel-light: #B8BFC7;
    --bg: #1A1A1A;
    --card: #242424;
    --border: #383838;
    --text: #E8E8E8;
    --text-dim: #888;
    --green: #3DB87A;
    --red: #E04444;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── HEADER ── */
  header {
    background: var(--concrete);
    border-bottom: 3px solid var(--orange);
    padding: 0 32px;
    display: flex;
    align-items: center;
    gap: 16px;
    height: 72px;
    position: sticky;
    top: 0;
    z-index: 100;
  }
  .header-icon {
    width: 44px; height: 44px;
    background: var(--orange);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
  }
  .header-icon svg { width: 26px; height: 26px; fill: #fff; }
  header h1 {
    font-family: 'Oswald', sans-serif;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #fff;
  }
  header h1 span { color: var(--orange); }
  .header-badge {
    margin-left: auto;
    background: var(--concrete-light);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px;
    color: var(--steel-light);
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }

  /* ── HERO STRIPE ── */
  .hero-stripe {
    background: repeating-linear-gradient(
      -55deg,
      var(--orange) 0px, var(--orange) 16px,
      var(--orange-dark) 16px, var(--orange-dark) 32px
    );
    height: 6px;
    opacity: 0.8;
  }

  /* ── MAIN LAYOUT ── */
  main {
    max-width: 1200px;
    margin: 0 auto;
    padding: 40px 24px 60px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 28px;
  }
  @media (max-width: 860px) {
    main { grid-template-columns: 1fr; }
  }

  /* ── CARD ── */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }
  .card-header {
    background: var(--concrete-mid);
    border-bottom: 1px solid var(--border);
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .card-header .dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--orange);
    box-shadow: 0 0 8px var(--orange);
    flex-shrink: 0;
  }
  .card-title {
    font-family: 'Oswald', sans-serif;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--steel-light);
  }
  .card-body { padding: 24px; }

  /* ── DROP ZONE ── */
  #dropzone {
    border: 2px dashed var(--concrete-light);
    border-radius: 10px;
    padding: 48px 24px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    position: relative;
  }
  #dropzone:hover, #dropzone.dragover {
    border-color: var(--orange);
    background: rgba(245,130,10,0.05);
  }
  #dropzone input[type=file] {
    position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%;
  }
  .drop-icon {
    width: 64px; height: 64px;
    margin: 0 auto 16px;
    background: var(--concrete-light);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
  }
  .drop-icon svg { width: 32px; height: 32px; fill: var(--orange); }
  #dropzone p.label {
    font-size: 16px; font-weight: 600; color: var(--text); margin-bottom: 6px;
  }
  #dropzone p.sub { font-size: 13px; color: var(--text-dim); }

  /* ── PREVIEW ── */
  #preview-wrap {
    margin-top: 20px;
    display: none;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    position: relative;
  }
  #preview-wrap img {
    width: 100%; display: block; max-height: 300px; object-fit: contain;
    background: #111;
  }
  .preview-badge {
    position: absolute; top: 10px; left: 10px;
    background: rgba(0,0,0,0.7);
    border: 1px solid var(--orange);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    color: var(--orange);
    font-weight: 600;
  }

  /* ── BUTTON ── */
  #analyze-btn {
    margin-top: 20px;
    width: 100%;
    background: var(--orange);
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 14px;
    font-family: 'Oswald', sans-serif;
    font-size: 17px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center; gap: 10px;
    transition: background 0.2s, transform 0.1s;
  }
  #analyze-btn:hover:not(:disabled) { background: var(--orange-dark); }
  #analyze-btn:active:not(:disabled) { transform: scale(0.98); }
  #analyze-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  #analyze-btn svg { width: 20px; height: 20px; fill: #fff; }

  /* ── SPINNER ── */
  .spinner {
    width: 20px; height: 20px;
    border: 3px solid rgba(255,255,255,0.3);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── STATUS BAR ── */
  #status-bar {
    margin-top: 16px;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    display: none;
    align-items: center;
    gap: 8px;
  }
  #status-bar.ok  { background: rgba(61,184,122,0.12); border: 1px solid var(--green); color: var(--green); display:flex; }
  #status-bar.err { background: rgba(224,68,68,0.12);  border: 1px solid var(--red);   color: var(--red);   display:flex; }

  /* ── STATS ROW ── */
  #stats-row {
    display: none;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    margin-bottom: 16px;
  }
  .stat-card {
    background: var(--concrete-mid);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
  }
  .stat-card .val {
    font-family: 'Oswald', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: var(--orange);
    line-height: 1;
  }
  .stat-card .lbl {
    font-size: 12px;
    color: var(--text-dim);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  /* ── JSON OUTPUT ── */
  #json-output-wrap {
    display: none;
    flex-direction: column;
    height: 100%;
  }
  .json-toolbar {
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 12px;
  }
  .json-toolbar .tag {
    background: var(--concrete-light);
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    color: var(--steel-light);
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }
  .json-toolbar button {
    margin-left: auto;
    background: var(--concrete-mid);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--steel-light);
    padding: 5px 12px;
    font-size: 12px;
    cursor: pointer;
    display: flex; align-items: center; gap: 6px;
    transition: all 0.15s;
  }
  .json-toolbar button:hover { border-color: var(--orange); color: var(--orange); }
  .json-toolbar button svg { width: 14px; height: 14px; fill: currentColor; }

  #json-box {
    flex: 1;
    background: #111;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    overflow: auto;
    font-family: 'Courier New', monospace;
    font-size: 13px;
    line-height: 1.6;
    color: #C9D1D9;
    max-height: 480px;
    white-space: pre;
  }
  /* syntax colors */
  .j-key   { color: #79C0FF; }
  .j-str   { color: #A5D6FF; }
  .j-num   { color: #F8A950; }
  .j-bool  { color: #E4885C; }
  .j-null  { color: #888; }
  .j-punc  { color: #666; }

  /* ── EMPTY STATE ── */
  #empty-state {
    height: 100%;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 40px;
    color: var(--text-dim);
    text-align: center;
    gap: 12px;
  }
  #empty-state svg { width: 56px; height: 56px; fill: var(--concrete-light); }
  #empty-state p { font-size: 14px; line-height: 1.6; }

  /* ── FOOTER ── */
  footer {
    text-align: center;
    padding: 24px;
    font-size: 12px;
    color: var(--text-dim);
    border-top: 1px solid var(--border);
  }
  footer span { color: var(--orange); }
</style>
</head>
<body>

<header>
  <div class="header-icon">
    <svg viewBox="0 0 24 24"><path d="M3 3h18v2H3V3zm0 4h18v2H3V7zm0 4h10v2H3v-2zm0 4h10v2H3v-2zm12 0h6v6h-6v-6zm1 1v4h4v-4h-4z"/></svg>
  </div>
  <h1>План<span>Анализ</span></h1>
  <div class="header-badge">Экстрактор стен v1.0</div>
</header>

<div class="hero-stripe"></div>

<main>
  <!-- LEFT: Upload -->
  <div class="card">
    <div class="card-header">
      <div class="dot"></div>
      <span class="card-title">Загрузка схемы</span>
    </div>
    <div class="card-body">
      <div id="dropzone">
        <input type="file" id="file-input" accept="image/png,image/jpeg,image/bmp,image/tiff"/>
        <div class="drop-icon">
          <svg viewBox="0 0 24 24"><path d="M19.35 10.04A7.49 7.49 0 0 0 12 4C9.11 4 6.6 5.64 5.35 8.04A5.994 5.994 0 0 0 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM14 13v4h-4v-4H7l5-5 5 5h-3z"/></svg>
        </div>
        <p class="label">Перетащите файл сюда</p>
        <p class="sub">или нажмите для выбора · PNG, JPG, BMP, TIFF</p>
      </div>

      <div id="preview-wrap">
        <span class="preview-badge">СХЕМА</span>
        <img id="preview-img" src="" alt="preview"/>
      </div>

      <button id="analyze-btn" disabled>
        <svg viewBox="0 0 24 24"><path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 1-2-2V9m0 0h18"/></svg>
        Анализировать стены
      </button>

      <div id="status-bar"></div>
    </div>
  </div>

  <!-- RIGHT: Result -->
  <div class="card">
    <div class="card-header">
      <div class="dot"></div>
      <span class="card-title">Результат — JSON</span>
    </div>
    <div class="card-body" style="display:flex;flex-direction:column;height:calc(100% - 53px);">

      <div id="stats-row">
        <div class="stat-card">
          <div class="val" id="stat-walls">—</div>
          <div class="lbl">Сегментов стен</div>
        </div>
        <div class="stat-card">
          <div class="val" id="stat-file">—</div>
          <div class="lbl">Файл</div>
        </div>
      </div>

      <div id="empty-state">
        <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6zm-1 1.5L18.5 9H13V3.5zM8 13h8v1H8v-1zm0 3h5v1H8v-1zm0-6h2v1H8v-1z"/></svg>
        <p>Загрузите изображение схемы<br/>и нажмите «Анализировать стены»<br/>— здесь появится JSON с координатами</p>
      </div>

      <div id="json-output-wrap">
        <div class="json-toolbar">
          <span class="tag">application/json</span>
          <button id="copy-btn" onclick="copyJson()">
            <svg viewBox="0 0 24 24"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>
            Копировать
          </button>
          <button onclick="downloadJson()">
            <svg viewBox="0 0 24 24"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>
            Скачать
          </button>
        </div>
        <div id="json-box"></div>
      </div>

    </div>
  </div>
</main>

<footer>
  © 2025 <span>ПланАнализ</span> · Компьютерное зрение для строительной документации
</footer>

<script>
  let currentJson = null;
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const analyzeBtn = document.getElementById('analyze-btn');
  const previewWrap = document.getElementById('preview-wrap');
  const previewImg = document.getElementById('preview-img');
  const statusBar = document.getElementById('status-bar');
  const emptyState = document.getElementById('empty-state');
  const jsonOutputWrap = document.getElementById('json-output-wrap');
  const jsonBox = document.getElementById('json-box');
  const statsRow = document.getElementById('stats-row');

  /* drag & drop */
  dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', e => {
    e.preventDefault(); dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; handleFile(); }
  });
  fileInput.addEventListener('change', handleFile);

  function handleFile() {
    const file = fileInput.files[0];
    if (!file) return;
    previewImg.src = URL.createObjectURL(file);
    previewWrap.style.display = 'block';
    analyzeBtn.disabled = false;
    hideStatus();
  }

  analyzeBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) return;

    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<div class="spinner"></div> Обработка…';
    hideStatus();
    emptyState.style.display = 'none';
    jsonOutputWrap.style.display = 'none';
    statsRow.style.display = 'none';

    const form = new FormData();
    form.append('file', file);

    try {
      const res = await fetch('/api/v1/process', { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) {
        showStatus('err', data.detail || 'Ошибка сервера');
      } else {
        currentJson = data;
        renderResult(data);
        showStatus('ok', `Успешно обработано: обнаружено ${data.wall_count} сегментов стен`);
      }
    } catch (e) {
      showStatus('err', 'Ошибка соединения: ' + e.message);
    }

    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 1-2-2V9m0 0h18"/></svg> Анализировать стены';
  });

  function renderResult(data) {
    document.getElementById('stat-walls').textContent = data.wall_count.toLocaleString('ru');
    document.getElementById('stat-file').textContent = truncate(data.filename, 14);
    statsRow.style.display = 'grid';
    jsonBox.innerHTML = syntaxHighlight(JSON.stringify(data, null, 2));
    jsonOutputWrap.style.display = 'flex';
  }

  function truncate(s, n) { return s && s.length > n ? s.slice(0, n) + '…' : s; }

  function showStatus(type, msg) {
    statusBar.className = type;
    statusBar.innerHTML = (type === 'ok'
      ? '<svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>'
      : '<svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/></svg>'
    ) + ' ' + msg;
  }
  function hideStatus() { statusBar.className = ''; statusBar.innerHTML = ''; statusBar.style.display = 'none'; }

  function syntaxHighlight(json) {
    return json
      .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
      .replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, m => {
        if (/^"/.test(m)) {
          if (/:$/.test(m)) return '<span class="j-key">' + m + '</span>';
          return '<span class="j-str">' + m + '</span>';
        }
        if (/true|false/.test(m)) return '<span class="j-bool">' + m + '</span>';
        if (/null/.test(m)) return '<span class="j-null">' + m + '</span>';
        return '<span class="j-num">' + m + '</span>';
      });
  }

  function copyJson() {
    if (!currentJson) return;
    navigator.clipboard.writeText(JSON.stringify(currentJson, null, 2)).then(() => {
      const btn = document.getElementById('copy-btn');
      btn.textContent = '✓ Скопировано';
      setTimeout(() => { btn.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg> Копировать'; }, 2000);
    });
  }

  function downloadJson() {
    if (!currentJson) return;
    const blob = new Blob([JSON.stringify(currentJson, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = (currentJson.filename || 'result').replace(/\.[^.]+$/, '') + '_walls.json';
    a.click();
  }
</script>
</body>
</html>"""


def _process_image_bytes(data: bytes) -> list:
    arr = np.frombuffer(data, np.uint8)
    scheme = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if scheme is None:
        raise ValueError("Could not decode image. Make sure you uploaded a valid PNG or JPEG.")

    scheme = cv2.cvtColor(scheme, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(scheme, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    walls = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    coords = []
    edges = cv2.Canny(walls, 10, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    delta = 10
    for contour in contours:
        points = contour[:, 0, :]
        n_points = len(points)
        if n_points < 2:
            continue
        for i in range(n_points - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            dist = float(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))
            if dist >= delta:
                coords.append([[int(x0), int(y0)], [int(x1), int(y1)]])
        if n_points > 2:
            x0, y0 = points[-1]
            x1, y1 = points[0]
            dist = float(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))
            if dist >= delta:
                coords.append([[int(x0), int(y0)], [int(x1), int(y1)]])
    return coords


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return HTMLResponse(content=HTML_PAGE)


@app.get("/health", summary="Health check")
def health_check():
    return {"status": "ok"}


@app.post(
    "/api/v1/process",
    summary="Process a floor plan image",
    description=(
        "Upload a floor plan image (PNG, JPG, BMP, TIFF) and receive a list of "
        "detected wall segments as pairs of [x, y] coordinate points."
    ),
)
async def process_floor_plan(file: UploadFile = File(...)):
    allowed = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/tiff"}
    if file.content_type and file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Upload a PNG or JPEG.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        coords = _process_image_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    walls = [{"id": f"w{i + 1}", "points": pts} for i, pts in enumerate(coords)]
    return JSONResponse(
        content={
            "filename": file.filename,
            "wall_count": len(walls),
            "walls": walls,
        }
    )
