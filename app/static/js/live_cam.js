/**
 * Road Damage AI — Live Camera Detection Engine
 * ================================================
 * Captures browser webcam frames, sends them to /api/live-frame every 2s,
 * draws detection boxes on a canvas overlay, and updates live stats.
 */

const CAM = (() => {
  // ── DOM refs (assigned on init) ──────────────────────────────────────
  let video, canvas, overlay, ctx, overlayCtx;
  let statTotal, statFrameCount, statDamagePct, statConsensus;
  let startBtn, stopBtn, resetBtn, statusIndicator, statusText;

  // ── State ─────────────────────────────────────────────────────────────
  let stream = null;
  let intervalId = null;
  let isRunning = false;
  let sessionTotal = 0;
  let frameCount = 0;
  let sessionConsensus = 0;
  const INTERVAL_MS = 2000; // send a frame every 2 seconds

  // Model colors (matches detector.py registry)
  const MODEL_COLORS = {
    model_a: "#ff3232",
    model_b: "#32dc32",
    model_c: "#3264ff",
  };

  // ── Init ─────────────────────────────────────────────────────────────
  function init() {
    video           = document.getElementById("cam-video");
    canvas          = document.getElementById("cam-capture");   // hidden capture canvas
    overlay         = document.getElementById("cam-overlay");   // visible annotation canvas
    ctx             = canvas.getContext("2d");
    overlayCtx      = overlay.getContext("2d");

    statTotal        = document.getElementById("stat-total");
    statFrameCount   = document.getElementById("stat-frames");
    statDamagePct    = document.getElementById("stat-damage-pct");
    statConsensus    = document.getElementById("stat-consensus");

    startBtn         = document.getElementById("cam-start");
    stopBtn          = document.getElementById("cam-stop");
    resetBtn         = document.getElementById("cam-reset");
    statusIndicator  = document.getElementById("cam-status-dot");
    statusText       = document.getElementById("cam-status-text");

    startBtn.addEventListener("click", startCamera);
    stopBtn.addEventListener("click", stopCamera);
    resetBtn.addEventListener("click", resetSession);
  }

  // ── Camera Control ────────────────────────────────────────────────────
  async function startCamera() {
    if (isRunning) return;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: 640, height: 480 },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();

      // Size overlay canvas to match video
      video.addEventListener("loadedmetadata", () => {
        overlay.width  = video.videoWidth;
        overlay.height = video.videoHeight;
        canvas.width   = video.videoWidth;
        canvas.height  = video.videoHeight;
      }, { once: true });

      setStatus(true);
      isRunning = true;
      startBtn.disabled = true;
      stopBtn.disabled  = false;

      // Start sending frames
      intervalId = setInterval(captureAndSend, INTERVAL_MS);
      // Send first frame immediately
      captureAndSend();

    } catch (err) {
      setStatusError(`Camera error: ${err.message}`);
    }
  }

  function stopCamera() {
    if (!isRunning) return;
    clearInterval(intervalId);
    intervalId = null;
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
    video.srcObject = null;
    isRunning = false;
    setStatus(false);
    startBtn.disabled = false;
    stopBtn.disabled  = true;
    // Clear overlay
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  }

  function resetSession() {
    sessionTotal     = 0;
    frameCount       = 0;
    sessionConsensus = 0;
    updateStats(0, 0, 0);
  }

  // ── Frame Capture + Send ──────────────────────────────────────────────
  function captureAndSend() {
    if (!isRunning || !video.videoWidth) return;

    // Draw current video frame onto hidden canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL("image/jpeg", 0.80);

    fetch("/api/live-frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame: dataURL }),
    })
      .then(r => r.json())
      .then(data => {
        if (data.error) {
          console.error("Live frame error:", data.error);
          return;
        }
        frameCount++;
        sessionTotal    += data.count || 0;
        sessionConsensus += data.consensus_detections || 0;
        const pct = data.damage_pct || 0;
        updateStats(data.count || 0, pct, data.consensus_detections || 0);
        drawDetections(data.detections || []);
      })
      .catch(err => console.error("Frame send error:", err));
  }

  // ── Draw Boxes on Overlay Canvas ──────────────────────────────────────
  function drawDetections(detections) {
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

    detections.forEach(det => {
      const [x1, y1, x2, y2] = det.bbox;
      const color = det.color_hex || MODEL_COLORS[det.model_id] || "#ffffff";
      const consensus = det.consensus || 1;
      const lineWidth = consensus > 1 ? 3 : 2;

      overlayCtx.strokeStyle = color;
      overlayCtx.lineWidth   = lineWidth;
      overlayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      // Label background
      const label = `${det.model_id.slice(-1).toUpperCase()}: ${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
      overlayCtx.font = "bold 13px Inter, sans-serif";
      const textW = overlayCtx.measureText(label).width;
      const lx = x1;
      const ly = Math.max(y1 - 6, 18);

      overlayCtx.fillStyle = color;
      overlayCtx.fillRect(lx, ly - 16, textW + 8, 20);
      overlayCtx.fillStyle = "#ffffff";
      overlayCtx.fillText(label, lx + 4, ly - 2);

      // Consensus star
      if (consensus > 1) {
        overlayCtx.fillStyle = "#f59e0b";
        overlayCtx.font = "bold 14px sans-serif";
        overlayCtx.fillText("★", x2 - 18, y1 + 16);
      }
    });
  }

  // ── Stats Update ──────────────────────────────────────────────────────
  function updateStats(frameDetections, damagePct, frameConsensus) {
    if (statTotal)      statTotal.textContent      = sessionTotal;
    if (statFrameCount) statFrameCount.textContent  = frameCount;
    if (statDamagePct)  statDamagePct.textContent   = damagePct.toFixed(1) + "%";
    if (statConsensus)  statConsensus.textContent   = sessionConsensus;
  }

  // ── Status Indicator ──────────────────────────────────────────────────
  function setStatus(running) {
    if (statusIndicator) {
      statusIndicator.className = running
        ? "status-dot live"
        : "status-dot idle";
    }
    if (statusText) {
      statusText.textContent = running ? "Live Detection Active" : "Camera Stopped";
    }
  }

  function setStatusError(msg) {
    if (statusText) statusText.textContent = msg;
    if (statusIndicator) statusIndicator.className = "status-dot error";
  }

  // ── Public ────────────────────────────────────────────────────────────
  return { init };
})();

document.addEventListener("DOMContentLoaded", CAM.init);
