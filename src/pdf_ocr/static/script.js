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

// Text-only overrides the output format, so gray out the dropdown when it's on.
const textOnlyToggleEl = document.getElementById('text-only-toggle');
const formatSelectEl = document.getElementById('format-select');
if (textOnlyToggleEl && formatSelectEl) {
    textOnlyToggleEl.addEventListener('change', () => {
        formatSelectEl.disabled = textOnlyToggleEl.checked;
    });
}

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

// Selecting files no longer auto-starts — it arms the "Run OCR" button.
function selectFiles(fileList) {
    const pdfs = Array.from(fileList).filter(f => f.type === 'application/pdf');
    if (pdfs.length === 0) {
        alert('Please choose PDF file(s).');
        return;
    }
    selectedFiles = pdfs;
    if (selectedFileEl) {
        selectedFileEl.innerText = pdfs.length === 1
            ? pdfs[0].name
            : `${pdfs.length} PDFs selected`;
    }
    if (readyRow) readyRow.classList.remove('hidden');
}

if (startBtn) {
    startBtn.addEventListener('click', () => {
        if (selectedFiles.length) processFiles(selectedFiles);
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

// Process a single file via /process and return its result descriptor.
async function processOne(file) {
    updateProgress(`${currentFileLabel}Uploading…`, 0);
    startTimer();

    const textOnly = textOnlyToggleEl ? textOnlyToggleEl.checked : false;
    // Text-only always produces a plain-text dump regardless of the dropdown.
    const format = textOnly ? 'txt' : (formatSelectEl ? formatSelectEl.value : 'pdf');
    const formatSuffix = { pdf: '.pdf', html: '.html', md: '.md', txt: '.txt' }[format] || '.pdf';

    const formData = new FormData();
    formData.append('file', file);
    formData.append('client_id', clientId);
    formData.append('format', format);
    formData.append('text_only', textOnly ? 'true' : 'false');

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
