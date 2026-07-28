// ---------------------------------------------------------------------------
// Config — point these at your backend
// ---------------------------------------------------------------------------
const API_BASE = "http://localhost:8000";
const WS_BASE = "ws://localhost:8000";
const TARGET_SAMPLE_RATE = 16000;
const CHUNK_SECONDS = 3; // must match backend/config.py CHUNK_SECONDS

// ---------------------------------------------------------------------------
// WAV encoding (used only for the short enrollment sample upload)
// ---------------------------------------------------------------------------
function encodeWAV(int16Samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + int16Samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + int16Samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, int16Samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < int16Samples.length; i++, offset += 2) {
    view.setInt16(offset, int16Samples[i], true);
  }

  return new Blob([view], { type: "audio/wav" });
}

function float32ToInt16(float32Array) {
  const int16 = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return int16;
}

// ---------------------------------------------------------------------------
// Shared mic capture helper. Runs a callback with each Float32 audio block
// at TARGET_SAMPLE_RATE. Returns a stop() function.
// ---------------------------------------------------------------------------
async function startMicCapture(onAudioBlock) {
  if (!window.isSecureContext) {
    throw new Error(
      "Mic access requires a secure context. Open this page via http://localhost:5500 " +
      "or http://127.0.0.1:5500 — not 0.0.0.0 or a bare IP address."
    );
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("This browser doesn't expose microphone access on this page (insecure origin?).");
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    if (err.name === "NotAllowedError") {
      throw new Error("Mic permission was denied. Click the lock/camera icon in the address bar and allow microphone access.");
    }
    if (err.name === "NotFoundError") {
      throw new Error("No microphone was found on this device.");
    }
    throw new Error(`Mic access failed: ${err.message}`);
  }
  // Requesting the AudioContext at 16kHz directly makes the browser resample
  // for us, avoiding manual downsampling code.
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)({
    sampleRate: TARGET_SAMPLE_RATE,
  });
  const source = audioCtx.createMediaStreamSource(stream);
  const processor = audioCtx.createScriptProcessor(4096, 1, 1);

  source.connect(processor);
  processor.connect(audioCtx.destination); // required by some browsers to keep the node alive
  processor.onaudioprocess = (e) => {
    const channelData = e.inputBuffer.getChannelData(0);
    onAudioBlock(new Float32Array(channelData)); // copy — buffer is reused internally
  };

  const stop = () => {
    processor.disconnect();
    source.disconnect();
    stream.getTracks().forEach((t) => t.stop());
    audioCtx.close();
  };

  return { stop, audioCtx };
}

// ---------------------------------------------------------------------------
// 1. ENROLLMENT
// ---------------------------------------------------------------------------
const enrollBtn = document.getElementById("enroll-record-btn");
const enrollNameInput = document.getElementById("enroll-name");
const enrollStatus = document.getElementById("enroll-status");
const speakerListEl = document.getElementById("speaker-list");

async function refreshSpeakerList() {
  try {
    const res = await fetch(`${API_BASE}/speakers`);
    const data = await res.json();
    speakerListEl.textContent = data.speakers.length ? data.speakers.join(", ") : "none yet";
  } catch (e) {
    speakerListEl.textContent = "(backend not reachable)";
  }
}

enrollBtn.addEventListener("click", async () => {
  const name = enrollNameInput.value.trim();
  if (!name) {
    enrollStatus.textContent = "Enter a name first.";
    return;
  }

  enrollStatus.textContent = "Recording... speak now (6s)";
  enrollBtn.disabled = true;

  const collected = [];
  let stop;
  try {
    const capture = await startMicCapture((block) => collected.push(float32ToInt16(block)));
    stop = capture.stop;
  } catch (err) {
    enrollStatus.textContent = err.message;
    enrollBtn.disabled = false;
    return;
  }

  setTimeout(async () => {
    stop();
    const totalLength = collected.reduce((sum, arr) => sum + arr.length, 0);
    const merged = new Int16Array(totalLength);
    let offset = 0;
    for (const arr of collected) {
      merged.set(arr, offset);
      offset += arr.length;
    }

    const wavBlob = encodeWAV(merged, TARGET_SAMPLE_RATE);
    const formData = new FormData();
    formData.append("name", name);
    formData.append("audio", wavBlob, `${name}.wav`);

    enrollStatus.textContent = "Uploading...";
    try {
      const res = await fetch(`${API_BASE}/enroll`, { method: "POST", body: formData });
      if (!res.ok) throw new Error(await res.text());
      enrollStatus.textContent = `Enrolled "${name}".`;
      enrollNameInput.value = "";
      refreshSpeakerList();
    } catch (e) {
      enrollStatus.textContent = "Enrollment failed — is the backend running?";
    }
    enrollBtn.disabled = false;
  }, 6000);
});

// ---------------------------------------------------------------------------
// 2. LIVE MEETING
// ---------------------------------------------------------------------------
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const meetingStatus = document.getElementById("meeting-status");
const transcriptEl = document.getElementById("transcript");
const summaryPanel = document.getElementById("summary-panel");
const summaryEl = document.getElementById("summary");

let ws = null;
let micStop = null;
let currentSessionId = null;
let sendBuffer = [];
let sendBufferLength = 0;
const CHUNK_SAMPLES = TARGET_SAMPLE_RATE * CHUNK_SECONDS;

function appendTranscriptEntry(entry) {
  const div = document.createElement("div");
  div.className = "entry";

  const known = entry.identified_as !== "Unknown";
  const speakerLabel = known ? entry.identified_as : entry.speaker_label;
  const speakerClass = known ? "known" : "unknown";

  div.innerHTML = `
    <span class="time">${entry.start_sec}s–${entry.end_sec}s</span>
    <span class="speaker ${speakerClass}">${speakerLabel} (${entry.confidence})</span>
    <span class="text">${entry.text}</span>
  `;
  transcriptEl.appendChild(div);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

startBtn.addEventListener("click", async () => {
  currentSessionId = crypto.randomUUID();
  transcriptEl.innerHTML = "";
  summaryPanel.style.display = "none";
  meetingStatus.textContent = "connecting...";

  ws = new WebSocket(`${WS_BASE}/ws/meeting/${currentSessionId}`);

  ws.onopen = async () => {
    meetingStatus.textContent = "recording";
    startBtn.disabled = true;
    stopBtn.disabled = false;

    sendBuffer = [];
    sendBufferLength = 0;

    try {
      const capture = await startMicCapture((block) => {
        const int16 = float32ToInt16(block);
        sendBuffer.push(int16);
        sendBufferLength += int16.length;

        if (sendBufferLength >= CHUNK_SAMPLES) {
          const merged = new Int16Array(sendBufferLength);
          let offset = 0;
          for (const arr of sendBuffer) {
            merged.set(arr, offset);
            offset += arr.length;
          }
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(merged.buffer);
          }
          sendBuffer = [];
          sendBufferLength = 0;
        }
      });
      micStop = capture.stop;
    } catch (err) {
      meetingStatus.textContent = err.message;
      startBtn.disabled = false;
      stopBtn.disabled = true;
      if (ws) ws.close();
      return;
    }
  };

  ws.onmessage = (event) => {
    const entry = JSON.parse(event.data);
    appendTranscriptEntry(entry);
  };

  ws.onerror = () => {
    meetingStatus.textContent = "connection error — is the backend running?";
  };

  ws.onclose = () => {
    meetingStatus.textContent = "stopped";
  };
});

stopBtn.addEventListener("click", async () => {
  if (micStop) micStop();
  if (ws) ws.close();
  startBtn.disabled = false;
  stopBtn.disabled = true;

  if (currentSessionId) {
    try {
      const res = await fetch(`${API_BASE}/meeting/${currentSessionId}/transcript`);
      const data = await res.json();
      renderSummary(data);
    } catch (e) {
      // backend may already be gone — non-fatal for the demo
    }
  }
});

function renderSummary(data) {
  const rows = Object.entries(data.speaking_time_seconds)
    .map(([speaker, seconds]) => `<tr><td>${speaker}</td><td>${seconds}s</td></tr>`)
    .join("");

  summaryEl.innerHTML = `
    <table>
      <tr><th>Speaker</th><th>Speaking time</th></tr>
      ${rows || '<tr><td colspan="2">No speech detected.</td></tr>'}
    </table>
  `;
  summaryPanel.style.display = "block";
}

// Initial load
refreshSpeakerList();
