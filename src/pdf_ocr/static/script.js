const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadView = document.getElementById('upload-view');
const processView = document.getElementById('process-view');
const resultView = document.getElementById('result-view');
const downloadBtn = document.getElementById('download-btn');
const resetBtn = document.getElementById('reset-btn');
const viewTextBtn = document.getElementById('view-text-btn');
const textPreview = document.getElementById('text-preview');
const textContent = document.getElementById('text-content');
const closePreview = document.getElementById('close-preview');

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

// Initialize WebSocket
let ws;

function connectWS() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/${clientId}`);
    
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.status) {
                updateProgress(data.status, data.percent);
            }
        } catch (e) {
            console.log("WS content is not JSON:", event.data);
        }
    };
    
    ws.onclose = () => {
        console.log("WS Disconnected");
    };
}

// Connect immediately or lazy? Lazy is better but easier to just connect processing.
// Let's connect when page loads or before upload.
connectWS();

function updateProgress(message, percent) {
    statusText.innerText = message;
    progressBar.style.width = `${percent}%`;
    subStatus.innerText = `${percent}% Complete`;
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
        handleFile(e.dataTransfer.files[0]);
    }
});

dropZone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

async function handleFile(file) {
    if (file.type !== 'application/pdf') {
        alert('Please upload a PDF file.');
        return;
    }

    // Switch UI
    uploadView.classList.add('hidden');
    processView.classList.remove('hidden');
    updateProgress("Uploading...", 0);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('client_id', clientId); // Send ID so server knows who to push progress to

    try {
        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'Processing failed');
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        
        // Setup Download
        downloadBtn.onclick = () => {
            const a = document.createElement('a');
            a.href = url;
            a.download = `OCR_${file.name}`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        };
        
        // Setup View Text
        viewTextBtn.onclick = async () => {
            try {
                const textResp = await fetch(`/text/${clientId}?t=${Date.now()}`);
                if (!textResp.ok) throw new Error("Could not fetch text");
                const textMap = await textResp.json();
                
                let html = "";
                for (const [page, lines] of Object.entries(textMap)) {
                    html += `<div style="margin-bottom: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 1rem;">\n`;
                    html += `<strong>Page ${parseInt(page) + 1}</strong>\n\n`;
                    html += lines.join('\n');
                    html += `</div>\n`;
                }
                textContent.innerHTML = html;
                textPreview.classList.remove('hidden');
            } catch (e) {
                alert("Text not available yet or error fetching it.");
            }
        };
        
        closePreview.onclick = () => {
            textPreview.classList.add('hidden');
        };

        // Setup Copy Button
        const copyBtn = document.getElementById('copy-text-btn');
        copyBtn.onclick = () => {
            const text = textContent.innerText;
            navigator.clipboard.writeText(text).then(() => {
                const originalContent = copyBtn.innerHTML;
                copyBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg> Copied!`;
                copyBtn.classList.add('bg-green-500/20', 'text-green-500');
                
                setTimeout(() => {
                    copyBtn.innerHTML = originalContent;
                    copyBtn.classList.remove('bg-green-500/20', 'text-green-500');
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy text: ', err);
            });
        };

        // Show Result
        processView.classList.add('hidden');
        resultView.classList.remove('hidden');


    } catch (error) {
        console.error(error);
        alert(`Error: ${error.message}`);
        resetUI();
    }
}

resetBtn.addEventListener('click', resetUI);

function resetUI() {
    fileInput.value = '';
    resultView.classList.add('hidden');
    processView.classList.add('hidden');
    uploadView.classList.remove('hidden');
    // Reset bar
    updateProgress("Ready", 0);
}

