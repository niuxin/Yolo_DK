/**
 * 无人机病害检测可视化界面前端逻辑。
 *
 * 三块职责：
 * 1. 左侧：通过 MJPEG 端点显示实时视频流（非抽帧）。
 * 2. 右上：显示最新病害代表帧，并按记录 bbox 叠加检测框。
 * 3. 右下：分页展示历史病害记录，点击行可回看该条代表帧。
 */
(function () {
  "use strict";

  const state = {
    pageSize: 50,
    offset: 0,
    total: 0,
    sessionId: "",
    previewSession: "",
    previewLastFrames: null,
    previewTimer: null,
    socket: null,
    socketRetry: null,
    activeRecordId: null,
    currentRecord: null,
    autoStarted: false,
  };

  const el = {
    dbStatus: document.getElementById("db-status"),
    wsStatus: document.getElementById("ws-status"),
    fpsStatus: document.getElementById("fps-status"),
    sourceInput: document.getElementById("source-input"),
    sessionInput: document.getElementById("session-input"),
    previewBtn: document.getElementById("preview-btn"),
    previewStopBtn: document.getElementById("preview-stop-btn"),
    previewImg: document.getElementById("preview-img"),
    previewPlaceholder: document.getElementById("preview-placeholder"),
    previewInfo: document.getElementById("preview-info"),
    detectCanvas: document.getElementById("detect-canvas"),
    detectPlaceholder: document.getElementById("detect-placeholder"),
    detectInfo: document.getElementById("detect-info"),
    sessionFilter: document.getElementById("session-filter"),
    refreshBtn: document.getElementById("refresh-btn"),
    recordsBody: document.getElementById("records-body"),
    recordsInfo: document.getElementById("records-info"),
    loadMoreBtn: document.getElementById("load-more-btn"),
  };

  const ctx = el.detectCanvas.getContext("2d");

  /* ---------------- 通用工具 ---------------- */

  function setBadge(node, text, kind) {
    node.textContent = text;
    node.className = "badge" + (kind ? " badge-" + kind : "");
  }

  function formatTime(value) {
    if (!value) return "-";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    const pad = (n) => String(n).padStart(2, "0");
    return (
      date.getFullYear() + "-" + pad(date.getMonth() + 1) + "-" + pad(date.getDate()) +
      " " + pad(date.getHours()) + ":" + pad(date.getMinutes()) + ":" + pad(date.getSeconds())
    );
  }

  async function fetchJson(url, options) {
    const response = await fetch(url, options);
    let body = null;
    try {
      body = await response.json();
    } catch (err) {
      body = null;
    }
    if (!response.ok) {
      const detail = body && body.detail ? body.detail : response.statusText;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return body;
  }

  /* ---------------- 健康状态 ---------------- */

  async function refreshHealth() {
    try {
      const health = await fetchJson("/api/health");
      if (health.database && health.database.available) {
        setBadge(el.dbStatus, "病害库已连接", "ok");
      } else {
        setBadge(el.dbStatus, "病害库未找到", "err");
      }
      applyDefaultSource(health.default_source);
    } catch (err) {
      setBadge(el.dbStatus, "界面服务异常", "err");
    }
  }

  /**
   * 用配置里的默认视频源预填输入框；首次拿到时自动开始预览。
   * 只在用户尚未手动填写、且尚未自动启动过时生效。
   */
  function applyDefaultSource(source) {
    if (!source) return;
    if (!el.sourceInput.value.trim()) {
      el.sourceInput.value = source;
    }
    if (state.autoStarted) return;
    if (state.previewSession) return;
    if (el.sourceInput.value.trim() !== source) return;
    state.autoStarted = true;
    openPreview();
  }

  /* ---------------- 左侧：实时预览 ---------------- */

  function startFpsCounter() {
    stopFpsCounter();
    state.previewLastFrames = null;
    // MJPEG 的 <img> 只在首帧触发一次 load，无法据此统计帧率，
    // 改为轮询服务端统计的累计帧数并计算差值。
    state.previewTimer = setInterval(async () => {
      if (!state.previewSession) return;
      try {
        const health = await fetchJson("/api/health");
        const streams = (health.preview && health.preview.streams) || [];
        const stream = streams.find((item) => item.session_id === state.previewSession);
        if (!stream) return;
        const total = Number(stream.frames || 0);
        if (state.previewLastFrames !== null) {
          const delta = total - state.previewLastFrames;
          setBadge(el.fpsStatus, "预览 " + Math.max(0, delta) + " fps");
        }
        state.previewLastFrames = total;
        if (stream.error) {
          el.previewInfo.textContent = "预览异常：" + stream.error;
        }
      } catch (err) {
        /* 统计失败不影响画面 */
      }
    }, 1000);
  }

  function stopFpsCounter() {
    if (state.previewTimer) {
      clearInterval(state.previewTimer);
      state.previewTimer = null;
    }
    state.previewLastFrames = null;
    setBadge(el.fpsStatus, "预览 -- fps");
  }

  async function openPreview() {
    const source = el.sourceInput.value.trim();
    const sessionId = el.sessionInput.value.trim() || "preview";
    if (!source) {
      el.previewInfo.textContent = "请先填写视频源地址";
      return;
    }
    el.previewBtn.disabled = true;
    el.previewInfo.textContent = "正在连接视频源…";
    try {
      const result = await fetchJson("/api/preview/open", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, source: source }),
      });
      state.previewSession = result.session_id;
      el.previewImg.src = result.url + "?t=" + Date.now();
      el.previewImg.hidden = false;
      el.previewPlaceholder.hidden = true;
      el.previewInfo.textContent = "预览中：" + result.source;
      startFpsCounter();
    } catch (err) {
      el.previewInfo.textContent = "预览启动失败：" + err.message;
      el.previewImg.hidden = true;
      el.previewPlaceholder.hidden = false;
    } finally {
      el.previewBtn.disabled = false;
    }
  }

  async function stopPreview() {
    const sessionId = state.previewSession;
    state.previewSession = "";
    el.previewImg.removeAttribute("src");
    el.previewImg.hidden = true;
    el.previewPlaceholder.hidden = false;
    el.previewInfo.textContent = "预览已停止";
    stopFpsCounter();
    if (!sessionId) return;
    try {
      await fetchJson("/api/preview/" + encodeURIComponent(sessionId), { method: "DELETE" });
    } catch (err) {
      /* 停止失败不影响界面 */
    }
  }

  el.previewImg.addEventListener("error", () => {
    if (!state.previewSession) return;
    el.previewInfo.textContent = "预览流中断，正在重连…";
  });

  /* ---------------- 右上：病害检测画面 ---------------- */

  function clearDetect() {
    ctx.clearRect(0, 0, el.detectCanvas.width, el.detectCanvas.height);
    el.detectCanvas.width = 0;
    el.detectCanvas.height = 0;
    el.detectCanvas.hidden = true;
    el.detectPlaceholder.hidden = false;
    el.detectInfo.textContent = "等待检测结果…";
    state.currentRecord = null;
  }

  function drawRecord(record) {
    if (!record || !record.image_url) return;
    state.currentRecord = record;
    const image = new Image();
    image.onload = () => {
      const width = image.naturalWidth || record.image_width || 1280;
      const height = image.naturalHeight || record.image_height || 720;
      el.detectCanvas.width = width;
      el.detectCanvas.height = height;
      el.detectCanvas.hidden = false;
      el.detectPlaceholder.hidden = true;
      ctx.drawImage(image, 0, 0, width, height);

      const bbox = record.bbox || {};
      const scaleX = width / (record.image_width || width);
      const scaleY = height / (record.image_height || height);
      const x1 = (bbox.x1 || 0) * scaleX;
      const y1 = (bbox.y1 || 0) * scaleY;
      const x2 = (bbox.x2 || 0) * scaleX;
      const y2 = (bbox.y2 || 0) * scaleY;
      const boxWidth = Math.max(1, x2 - x1);
      const boxHeight = Math.max(1, y2 - y1);

      ctx.lineWidth = Math.max(2, Math.round(width / 480));
      ctx.strokeStyle = "#3fb950";
      ctx.strokeRect(x1, y1, boxWidth, boxHeight);

      const label =
        (record.disease_type || "病害") +
        " " + (Number(record.confidence || 0) * 100).toFixed(1) + "%";
      ctx.font = Math.max(12, Math.round(width / 64)) + "px 'Segoe UI', sans-serif";
      const textWidth = ctx.measureText(label).width;
      const labelHeight = Math.max(18, Math.round(width / 48));
      const labelY = y1 - labelHeight < 0 ? y1 : y1 - labelHeight;
      ctx.fillStyle = "rgba(63, 185, 80, .85)";
      ctx.fillRect(x1, labelY, textWidth + 12, labelHeight);
      ctx.fillStyle = "#0f1419";
      ctx.textBaseline = "middle";
      ctx.fillText(label, x1 + 6, labelY + labelHeight / 2);

      el.detectInfo.textContent =
        "记录 #" + record.id + " · " + formatTime(record.captured_at) +
        " · 质量分 " + Number(record.quality_score || 0).toFixed(3);
    };
    image.onerror = () => {
      el.detectInfo.textContent = "代表帧图片加载失败";
    };
    image.src = record.image_url + "?t=" + Date.now();
  }

  /* ---------------- 右下：历史病害数据 ---------------- */

  function renderRows(records, append) {
    if (!append) {
      el.recordsBody.innerHTML = "";
    }
    if (!records.length && !append) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 7;
      cell.className = "empty";
      cell.textContent = "暂无数据";
      row.appendChild(cell);
      el.recordsBody.appendChild(row);
      return;
    }
    records.forEach((record) => {
      const row = document.createElement("tr");
      row.dataset.recordId = String(record.id);
      if (state.activeRecordId === record.id) row.classList.add("active");

      const cells = [
        String(record.id),
        formatTime(record.captured_at),
        record.disease_type || "-",
        (Number(record.confidence || 0) * 100).toFixed(1) + "%",
        Number(record.quality_score || 0).toFixed(3),
        record.session_id || "-",
        record.description || "",
      ];
      cells.forEach((text, index) => {
        const cell = document.createElement("td");
        cell.textContent = text;
        if (index === 6) {
          cell.className = "desc";
          cell.title = text;
        }
        row.appendChild(cell);
      });

      row.addEventListener("click", () => {
        state.activeRecordId = record.id;
        Array.from(el.recordsBody.children).forEach((item) => {
          item.classList.toggle("active", item.dataset.recordId === String(record.id));
        });
        drawRecord(record);
      });
      el.recordsBody.appendChild(row);
    });
  }

  function updateRecordsInfo() {
    el.recordsInfo.textContent = "共 " + state.total + " 条，已显示 " +
      el.recordsBody.querySelectorAll("tr[data-record-id]").length + " 条";
    el.loadMoreBtn.hidden = state.offset >= state.total;
  }

  async function loadRecords(reset) {
    if (reset) {
      state.offset = 0;
      state.activeRecordId = null;
    }
    const params = new URLSearchParams({
      limit: String(state.pageSize),
      offset: String(state.offset),
    });
    if (state.sessionId) params.set("session_id", state.sessionId);
    try {
      const data = await fetchJson("/api/records?" + params.toString());
      state.total = data.total || 0;
      renderRows(data.records || [], !reset);
      state.offset += (data.records || []).length;
      updateRecordsInfo();
    } catch (err) {
      el.recordsInfo.textContent = "加载失败：" + err.message;
    }
  }

  async function loadSessions() {
    try {
      const data = await fetchJson("/api/sessions?limit=50");
      const current = el.sessionFilter.value;
      el.sessionFilter.innerHTML = '<option value="">全部会话</option>';
      (data.sessions || []).forEach((session) => {
        const option = document.createElement("option");
        option.value = session.session_id;
        option.textContent = session.session_id + " (" + session.records + ")";
        el.sessionFilter.appendChild(option);
      });
      el.sessionFilter.value = current;
    } catch (err) {
      /* 会话列表失败不影响主流程 */
    }
  }

  /* ---------------- WebSocket 实时推送 ---------------- */

  function connectSocket() {
    if (state.socket && state.socket.readyState <= 1) return;
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(scheme + "://" + location.host + "/ws/events");
    state.socket = socket;

    socket.addEventListener("open", () => {
      setBadge(el.wsStatus, "实时通道已连接", "ok");
    });

    socket.addEventListener("message", (event) => {
      let payload = null;
      try {
        payload = JSON.parse(event.data);
      } catch (err) {
        return;
      }
      if (payload.type === "record" && payload.record) {
        handleIncomingRecord(payload.record);
      }
    });

    socket.addEventListener("close", () => {
      setBadge(el.wsStatus, "实时通道已断开", "warn");
      state.socket = null;
      if (state.socketRetry) clearTimeout(state.socketRetry);
      state.socketRetry = setTimeout(connectSocket, 3000);
    });

    socket.addEventListener("error", () => {
      setBadge(el.wsStatus, "实时通道异常", "err");
    });
  }

  function handleIncomingRecord(record) {
    if (state.sessionId && record.session_id !== state.sessionId) return;
    drawRecord(record);
    state.total += 1;
    const row = document.createElement("tr");
    row.dataset.recordId = String(record.id);
    const cells = [
      String(record.id),
      formatTime(record.captured_at),
      record.disease_type || "-",
      (Number(record.confidence || 0) * 100).toFixed(1) + "%",
      Number(record.quality_score || 0).toFixed(3),
      record.session_id || "-",
      record.description || "",
    ];
    cells.forEach((text, index) => {
      const cell = document.createElement("td");
      cell.textContent = text;
      if (index === 6) {
        cell.className = "desc";
        cell.title = text;
      }
      row.appendChild(cell);
    });
    row.addEventListener("click", () => {
      state.activeRecordId = record.id;
      Array.from(el.recordsBody.children).forEach((item) => {
        item.classList.toggle("active", item.dataset.recordId === String(record.id));
      });
      drawRecord(record);
    });
    const empty = el.recordsBody.querySelector("td.empty");
    if (empty) el.recordsBody.innerHTML = "";
    el.recordsBody.insertBefore(row, el.recordsBody.firstChild);
    updateRecordsInfo();
  }

  /* ---------------- 事件绑定 ---------------- */

  el.previewBtn.addEventListener("click", openPreview);
  el.previewStopBtn.addEventListener("click", stopPreview);
  el.refreshBtn.addEventListener("click", () => {
    loadRecords(true);
    loadSessions();
    refreshHealth();
  });
  el.loadMoreBtn.addEventListener("click", () => loadRecords(false));
  el.sessionFilter.addEventListener("change", () => {
    state.sessionId = el.sessionFilter.value;
    loadRecords(true);
  });
  el.sessionInput.addEventListener("change", () => {
    if (!el.sessionInput.value.trim()) return;
    if (!el.sourceInput.value.trim()) return;
  });

  /* ---------------- 启动 ---------------- */

  refreshHealth();
  loadRecords(true);
  loadSessions();
  connectSocket();
  setInterval(refreshHealth, 15000);
  setInterval(loadSessions, 30000);
})();
