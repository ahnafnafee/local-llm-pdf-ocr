const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadView = document.getElementById('upload-view');
const processView = document.getElementById('process-view');
const resultView = document.getElementById('result-view');
const resetBtn = document.getElementById('reset-btn');
const textPreview = document.getElementById('text-preview');
const textContent = document.getElementById('text-content');
const closePreview = document.getElementById('close-preview');

const startBtn = document.getElementById('start-btn');
const readyRow = document.getElementById('ready-row');
const selectedFileEl = document.getElementById('selected-file');
const elapsedTimeEl = document.getElementById('elapsed-time');
const resultList = document.getElementById('result-list');
const resultTitle = document.getElementById('result-title');
const resultSummary = document.getElementById('result-summary');
let selectedFiles = [];

const themeBtn = document.getElementById('theme-btn');
const moonIcon = document.getElementById('moon-icon');
const sunIcon = document.getElementById('sun-icon');

// Theme Logic
function setTheme(isDark) {
    document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
    moonIcon.classList.toggle('hidden', isDark);
    sunIcon.classList.toggle('hidden', !isDark);
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
}

themeBtn.addEventListener('click', () => {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    setTheme(!isDark);
});

// Init Theme
const savedTheme = localStorage.getItem('theme') || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
setTheme(savedTheme === 'dark');


// Progress Elements
const progressBar = document.getElementById('progress-bar');
const statusText = document.getElementById('status-text');
const subStatus = document.getElementById('sub-status');

// Generate Client ID
const clientId = Math.random().toString(36).substring(7);

// Prefix shown on every progress line so multi-file runs say which file.
let currentFileLabel = '';

// Initialize WebSocket
let ws;

function connectWS() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/${clientId}`);

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.status) {
                updateProgress(currentFileLabel + data.status, data.percent);
            }
        } catch (e) {
            console.log("WS content is not JSON:", event.data);
        }
    };

    ws.onclose = () => {
        console.log("WS Disconnected");
    };
}

connectWS();

function updateProgress(message, percent) {
    statusText.innerText = message;
    progressBar.style.width = `${percent}%`;
    subStatus.innerText = `${percent}%`;
}

// Elapsed-time stopwatch: ticks during processing, frozen per result row.
let timerInterval = null;
let startTime = 0;

function formatDuration(seconds) {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return `${m}m ${String(s).padStart(2, '0')}s`;
}

function startTimer() {
    startTime = performance.now();
    if (elapsedTimeEl) elapsedTimeEl.innerText = '0.0s';
    timerInterval = setInterval(() => {
        if (elapsedTimeEl) {
            elapsedTimeEl.innerText = formatDuration((performance.now() - startTime) / 1000);
        }
    }, 100);
}

function stopTimer() {
    if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
    return (performance.now() - startTime) / 1000;
}

// ============================================================================
// Configuration surface: engine, model, output, advanced options.
// ============================================================================

const ENGINE_DESC = {
    hybrid: 'Surya layout detection + a full-page vision LLM + DP line-to-box alignment, with per-box crop refine. Works with any OCR-capable VLM.',
    grounded: 'A bbox-native VLM (Qwen3-VL, Qwen2.5-VL, …) returns text and coordinates in one call. Skips Surya, alignment, and refine — point Model at a grounding-capable VLM.',
    text: 'OCR each page to plain text — no layout detection, no alignment, no refine, no detection-model load. The fastest path; raise Concurrency to read more pages at once.',
};

const engineGroup = document.getElementById('engine-group');
const engineDesc = document.getElementById('engine-desc');
const formatSelectEl = document.getElementById('format-select');
const formatNote = document.getElementById('format-note');
const modelInput = document.getElementById('model-input');
const modelList = document.getElementById('model-list');
const modelStatus = document.getElementById('model-status');
const modelDot = document.getElementById('model-dot');
const refreshModelsBtn = document.getElementById('refresh-models');
const apiBaseInput = document.getElementById('opt-api-base');

// Every persisted control, keyed by the localStorage field name.
const OPT_FIELDS = {
    engine: null, // handled specially (segmented control)
    format: formatSelectEl,
    model: modelInput,
    api_base: apiBaseInput,
    dpi: document.getElementById('opt-dpi'),
    pages: document.getElementById('opt-pages'),
    concurrency: document.getElementById('opt-concurrency'),
    max_image_dim: document.getElementById('opt-max-dim'),
    verify_model: document.getElementById('opt-verify'),
    refine: document.getElementById('opt-refine'),
    dense_mode: document.getElementById('opt-dense-mode'),
    dense_threshold: document.getElementById('opt-dense-threshold'),
    preprocess: document.getElementById('opt-preprocess'),
    min_box_confidence: document.getElementById('opt-min-conf'),
    html_mode: document.getElementById('opt-html-mode'),
    html_invert_dark: document.getElementById('opt-invert-dark'),
    html_hover_text: document.getElementById('opt-hover-text'),
};

let engine = 'hybrid';

function applyVisibility() {
    document.querySelectorAll('.opt-group[data-engine]').forEach(el => {
        const forEngine = el.dataset.engine;
        el.classList.toggle('hidden', forEngine !== 'all' && forEngine !== engine);
    });
    document.querySelectorAll('.opt-group[data-format]').forEach(el => {
        el.classList.toggle('hidden', formatSelectEl.value !== el.dataset.format);
    });
}

function setEngine(next) {
    engine = next;
    engineGroup.querySelectorAll('button').forEach(b =>
        b.setAttribute('aria-pressed', String(b.dataset.engine === next)));
    engineDesc.textContent = ENGINE_DESC[next];
    if (next === 'text') {
        formatSelectEl.dataset.prev = formatSelectEl.value === 'txt' ? 'pdf' : formatSelectEl.value;
        formatSelectEl.value = 'txt';
        formatSelectEl.disabled = true;
        formatNote.textContent = 'text-only writes plain .txt';
    } else {
        formatSelectEl.disabled = false;
        formatNote.textContent = '';
        if (formatSelectEl.value === 'txt') formatSelectEl.value = formatSelectEl.dataset.prev || 'pdf';
    }
    applyVisibility();
}

engineGroup.querySelectorAll('button').forEach(btn => {
    btn.addEventListener('click', () => setEngine(btn.dataset.engine));
});
formatSelectEl.addEventListener('change', applyVisibility);

// The effective output format: text engine always dumps .txt.
function currentFormat() {
    return engine === 'text' ? 'txt' : formatSelectEl.value;
}

// --- model discovery --------------------------------------------------------

function shortEndpoint(url) {
    try { return new URL(url).host; } catch (e) { return url; }
}

function setModelStatus(text, ok) {
    modelStatus.textContent = text;
    modelDot.classList.toggle('off', !ok);
}

async function loadModels() {
    const apiBase = apiBaseInput.value.trim();
    const url = apiBase ? `/models?api_base=${encodeURIComponent(apiBase)}` : '/models';
    setModelStatus('Querying endpoint…', true);
    try {
        const resp = await fetch(url);
        const data = await resp.json();
        modelList.replaceChildren();
        for (const m of (data.models || [])) {
            const o = document.createElement('option');
            o.value = m;
            modelList.appendChild(o);
        }
        if (!modelInput.value) modelInput.value = data.default || '';
        const where = shortEndpoint(data.endpoint || apiBase || '');
        if (data.models && data.models.length) {
            setModelStatus(`${data.models.length} model(s) loaded · ${where}`, true);
        } else {
            setModelStatus(`No models listed · type a name · ${where}`, false);
        }
    } catch (e) {
        setModelStatus('Endpoint unreachable · type a model name', false);
    }
}

if (refreshModelsBtn) refreshModelsBtn.addEventListener('click', loadModels);
if (apiBaseInput) apiBaseInput.addEventListener('change', loadModels);

// --- persistence ------------------------------------------------------------

const CONFIG_KEY = 'ocr-config';

function saveConfig() {
    const cfg = { engine };
    for (const [name, el] of Object.entries(OPT_FIELDS)) {
        if (!el) continue;
        cfg[name] = el.type === 'checkbox' ? el.checked : el.value;
    }
    try { localStorage.setItem(CONFIG_KEY, JSON.stringify(cfg)); } catch (e) { /* quota */ }
}

function restoreConfig() {
    let cfg;
    try { cfg = JSON.parse(localStorage.getItem(CONFIG_KEY) || '{}'); } catch (e) { cfg = {}; }
    for (const [name, el] of Object.entries(OPT_FIELDS)) {
        if (!el || !(name in cfg)) continue;
        if (el.type === 'checkbox') el.checked = !!cfg[name];
        else el.value = cfg[name];
    }
    setEngine(cfg.engine || 'hybrid');
}

// Text-only overrides the output format, so gray out the dropdown when it's on.
// (Handled inside setEngine — this just wires the initial state.)
restoreConfig();
loadModels();

// Drag & Drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        selectFiles(e.dataTransfer.files);
    }
});

dropZone.addEventListener('click', () => {
    fileInput.click();
});

// The drop zone is a keyboard-reachable button.
dropZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fileInput.click();
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        selectFiles(e.target.files);
    }
});

// PDFs and images are both accepted. Type sniffing is unreliable for AVIF /
// TIFF across browsers, so fall back to the extension.
const ALLOWED_EXT = /\.(pdf|jpe?g|png|bmp|webp|tiff?|avif)$/i;
function isAllowedFile(f) {
    return f.type === 'application/pdf'
        || (f.type && f.type.startsWith('image/'))
        || ALLOWED_EXT.test(f.name);
}

// Selecting files no longer auto-starts — it arms the "Run OCR" button.
function selectFiles(fileList) {
    const files = Array.from(fileList).filter(isAllowedFile);
    if (files.length === 0) {
        alert('Please choose PDF or image file(s).');
        return;
    }
    selectedFiles = files;
    if (selectedFileEl) {
        selectedFileEl.innerText = files.length === 1
            ? files[0].name
            : `${files.length} files selected`;
    }
    if (readyRow) readyRow.classList.remove('hidden');
}

if (startBtn) {
    startBtn.addEventListener('click', () => {
        if (selectedFiles.length) {
            saveConfig();
            processFiles(selectedFiles);
        }
    });
}

async function processFiles(files) {
    if (!files.length) return;
    uploadView.classList.add('hidden');
    processView.classList.remove('hidden');

    const results = [];
    for (let i = 0; i < files.length; i++) {
        currentFileLabel = files.length > 1 ? `File ${i + 1}/${files.length}: ` : '';
        try {
            results.push(await processOne(files[i]));
        } catch (error) {
            console.error(error);
            stopTimer();
            // Record the failure and keep going with the remaining files.
            results.push({ name: files[i].name, error: error.message || 'failed' });
        }
    }
    currentFileLabel = '';

    renderResults(results, files.length);
    processView.classList.add('hidden');
    resultView.classList.remove('hidden');
}

const FORMAT_SUFFIX = { pdf: '.pdf', html: '.html', md: '.md', txt: '.txt' };

// Process a single file via /process and return its result descriptor.
async function processOne(file) {
    updateProgress(`${currentFileLabel}Uploading…`, 0);
    startTimer();

    const format = currentFormat();
    const formatSuffix = FORMAT_SUFFIX[format] || '.pdf';

    const formData = new FormData();
    formData.append('file', file);
    formData.append('client_id', clientId);
    formData.append('engine', engine);
    formData.append('format', format);
    formData.append('model', modelInput.value.trim());
    formData.append('api_base', apiBaseInput.value.trim());
    formData.append('dpi', OPT_FIELDS.dpi.value || '200');
    formData.append('pages', OPT_FIELDS.pages.value.trim());
    formData.append('concurrency', OPT_FIELDS.concurrency.value || '0');
    formData.append('max_image_dim', OPT_FIELDS.max_image_dim.value || '1024');
    formData.append('refine', OPT_FIELDS.refine.checked ? 'true' : 'false');
    formData.append('dense_mode', OPT_FIELDS.dense_mode.value);
    formData.append('dense_threshold', OPT_FIELDS.dense_threshold.value || '60');
    formData.append('preprocess', OPT_FIELDS.preprocess.value);
    formData.append('min_box_confidence', OPT_FIELDS.min_box_confidence.value.trim());
    formData.append('html_mode', OPT_FIELDS.html_mode.value);
    formData.append('html_invert_dark', OPT_FIELDS.html_invert_dark.checked ? 'true' : 'false');
    formData.append('html_hover_text', OPT_FIELDS.html_hover_text.checked ? 'true' : 'false');
    formData.append('verify_model', OPT_FIELDS.verify_model.checked ? 'true' : 'false');

    const response = await fetch('/process', { method: 'POST', body: formData });
    if (!response.ok) {
        let msg = 'Processing failed';
        try { msg = (await response.json()).error || msg; } catch (e) { /* non-JSON error */ }
        throw new Error(msg);
    }

    const blob = await response.blob();
    const secs = stopTimer();
    const url = window.URL.createObjectURL(blob);
    const baseName = file.name.replace(/\.[^.]+$/, '');
    const outName = `OCR_${baseName}${formatSuffix}`;

    // Stash this file's recognized text now — the next file's /process call
    // overwrites the server-side text_{client_id}.json.
    let textMap = null;
    try {
        const textResp = await fetch(`/text/${clientId}?t=${Date.now()}`);
        if (textResp.ok) textMap = await textResp.json();
    } catch (e) { /* text preview is best-effort */ }

    return { name: file.name, outName, url, secs, textMap };
}

function renderResults(results, total) {
    const ok = results.filter(r => !r.error).length;
    if (resultTitle) resultTitle.innerText = total > 1 ? `Done — ${ok}/${total} files` : 'Done';
    if (resultSummary) {
        const failed = results.length - ok;
        resultSummary.innerText = failed ? `${failed} file(s) failed` : 'Saved locally';
    }
    if (!resultList) return;
    resultList.replaceChildren();
    for (const r of results) resultList.appendChild(buildResultRow(r));
}

function buildResultRow(r) {
    const row = document.createElement('div');
    row.className = 'result-row';

    const info = document.createElement('div');
    info.className = 'result-row-info';
    const name = document.createElement('span');
    name.className = 'result-row-name';
    name.innerText = r.error ? r.name : r.outName;
    info.appendChild(name);
    const meta = document.createElement('span');
    meta.className = r.error ? 'result-row-time result-row-error' : 'result-row-time';
    meta.innerText = r.error ? `Error: ${r.error}` : formatDuration(r.secs);
    info.appendChild(meta);
    row.appendChild(info);

    if (r.error) return row;

    const actions = document.createElement('div');
    actions.className = 'result-row-actions';
    if (r.textMap) {
        const viewBtn = document.createElement('button');
        viewBtn.className = 'btn btn-secondary btn-compact';
        viewBtn.innerText = 'View text';
        viewBtn.onclick = () => openTextDrawer(r.textMap);
        actions.appendChild(viewBtn);
    }
    const dlBtn = document.createElement('button');
    dlBtn.className = 'btn btn-primary btn-compact';
    dlBtn.innerText = 'Download';
    dlBtn.onclick = () => {
        const a = document.createElement('a');
        a.href = r.url;
        a.download = r.outName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    };
    actions.appendChild(dlBtn);
    row.appendChild(actions);
    return row;
}

function openTextDrawer(textMap) {
    textContent.replaceChildren();
    for (const [page, lines] of Object.entries(textMap)) {
        const block = document.createElement('div');
        block.className = 'page-block';
        const label = document.createElement('span');
        label.className = 'page-label';
        label.innerText = `Page ${parseInt(page) + 1}`;
        block.appendChild(label);
        // textContent assignment keeps OCR output inert —
        // recognized text must never parse as markup.
        block.appendChild(document.createTextNode(lines.join('\n')));
        textContent.appendChild(block);
    }
    textPreview.classList.remove('hidden');
}

// Drawer controls (wired once — the drawer is shared across result rows).
if (closePreview) {
    closePreview.onclick = () => textPreview.classList.add('hidden');
}
const copyBtn = document.getElementById('copy-text-btn');
if (copyBtn) {
    copyBtn.onclick = () => {
        navigator.clipboard.writeText(textContent.innerText).then(() => {
            const original = copyBtn.innerHTML;
            copyBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg> Copied`;
            setTimeout(() => { copyBtn.innerHTML = original; }, 2000);
        }).catch(err => console.error('Failed to copy text: ', err));
    };
}

resetBtn.addEventListener('click', resetUI);

function resetUI() {
    stopTimer();
    currentFileLabel = '';
    selectedFiles = [];
    fileInput.value = '';
    if (readyRow) readyRow.classList.add('hidden');
    if (selectedFileEl) selectedFileEl.innerText = '';
    if (resultList) resultList.replaceChildren();
    if (elapsedTimeEl) elapsedTimeEl.innerText = '0.0s';
    resultView.classList.add('hidden');
    processView.classList.add('hidden');
    uploadView.classList.remove('hidden');
    updateProgress("Ready", 0);
}
