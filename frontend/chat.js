/* Comic assistant: a chat that knows the comic you just made and edits it on request.
   Talks to /api/comics/{id}/... (see backend/main.py). Everything user- or model-supplied is
   inserted with textContent, never innerHTML. */
(() => {
  const $ = (id) => document.getElementById(id);
  const ALLOWED = [".pdf", ".docx", ".png", ".jpg", ".jpeg", ".svg"];
  const MAX_FILES = 5;
  const MAX_MB = 15;
  const STORE_KEY = "comicforge.lastComic";
  const SUGGESTIONS = [
    "Make the dialogue funnier",
    "Redraw panel 1 from a different angle",
    "Add a plot-twist panel at the end",
    "Change the art style to watercolor",
    "Make the text bigger",
  ];

  let comicId = null;
  let busy = false;
  let pending = []; // File[] waiting to be sent with the next message

  const dock = $("chatDock"), panel = $("chatPanel"), toggle = $("chatToggle");
  const log = $("chatLog"), input = $("chatInput"), form = $("chatForm");
  const undoBtn = $("chatUndo"), redoBtn = $("chatRedo"), versionEl = $("chatVersion");
  const fileInput = $("chatFile"), pendingEl = $("chatPending"), suggestionsEl = $("chatSuggestions");

  const remember = (id) => { try { id ? localStorage.setItem(STORE_KEY, id) : localStorage.removeItem(STORE_KEY); } catch {} };
  const recall = () => { try { return localStorage.getItem(STORE_KEY); } catch { return null; } };

  // ------------------------------------------------------------------ rendering
  function scrollDown() { log.scrollTop = log.scrollHeight; }

  function chip(name, file) {
    const c = document.createElement("span");
    c.className = "chat-chip";
    if (file && file.type.startsWith("image/")) {
      const img = document.createElement("img");
      img.src = URL.createObjectURL(file);
      img.alt = "";
      img.onload = () => URL.revokeObjectURL(img.src);
      c.appendChild(img);
    }
    const label = document.createElement("span");
    label.textContent = name;
    label.title = name;
    c.appendChild(label);
    return c;
  }

  function addMsg(role, text, { kind = "", atts = [] } = {}) {
    const el = document.createElement("div");
    el.className = `chat-msg ${role} ${kind}`.trim();
    if (atts.length) {
      const wrap = document.createElement("div");
      wrap.className = "chat-atts";
      atts.forEach((a) => wrap.appendChild(chip(a.name, a.file)));
      el.appendChild(wrap);
    }
    if (text) {
      const body = document.createElement("div");
      body.className = "chat-text";
      body.textContent = text;
      el.appendChild(body);
    }
    log.appendChild(el);
    scrollDown();
    return el;
  }

  function addNote(text) {
    const n = document.createElement("div");
    n.className = "chat-note";
    n.textContent = text;
    log.appendChild(n);
    scrollDown();
  }

  function addTyping() {
    const el = document.createElement("div");
    el.className = "chat-msg assistant";
    el.setAttribute("role", "status");
    el.setAttribute("aria-label", "Assistant is thinking");
    el.innerHTML = '<span class="chat-typing"><i></i><i></i><i></i></span>';
    log.appendChild(el);
    scrollDown();
    return el;
  }

  function addProgress() {
    const el = document.createElement("div");
    el.className = "chat-msg assistant chat-progress";
    el.innerHTML = '<div class="progress-bar"><div class="progress-fill"></div></div><p></p>';
    log.appendChild(el);
    scrollDown();
    return {
      el,
      set(job) {
        el.querySelector(".progress-fill").style.width = `${Math.round((job.progress || 0) * 100)}%`;
        el.querySelector("p").textContent = job.message || "";
        scrollDown();
      },
    };
  }

  function renderHistory(entries) {
    log.textContent = "";
    if (!entries.length) {
      addMsg("assistant",
        "Your comic is ready! Tell me what to change — I can rewrite dialogue, redraw a panel, add, remove or reorder panels, " +
        "tweak a character or switch the art style. Attach a PDF, Word doc, PNG, JPG or SVG if you want me to work from a reference.");
    }
    entries.forEach((e) => {
      const kind = e.kind === "result" || e.kind === "error" ? e.kind : "";
      addMsg(e.role, e.content, { kind, atts: (e.attachments || []).map((a) => ({ name: a.name })) });
    });
  }

  function renderSuggestions() {
    suggestionsEl.textContent = "";
    if (log.querySelectorAll(".chat-msg.user").length) return;
    SUGGESTIONS.forEach((s) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = s;
      b.addEventListener("click", () => { input.value = s; autosize(); input.focus(); });
      suggestionsEl.appendChild(b);
    });
  }

  function renderPending() {
    pendingEl.textContent = "";
    pending.forEach((f, i) => {
      const c = chip(f.name, f);
      const x = document.createElement("button");
      x.type = "button";
      x.textContent = "✕";
      x.setAttribute("aria-label", `Remove ${f.name}`);
      x.addEventListener("click", () => { pending.splice(i, 1); renderPending(); });
      c.appendChild(x);
      pendingEl.appendChild(c);
    });
  }

  function autosize() {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
  }

  function setBusy(b) {
    busy = b;
    input.disabled = b;
    $("chatSend").disabled = b;
    $("chatAttach").disabled = b;
    undoBtn.disabled = b || !state.can_undo;
    redoBtn.disabled = b || !state.can_redo;
    if (!b) input.focus();
  }

  // ------------------------------------------------------------------ comic state
  let state = { can_undo: false, can_redo: false };

  function applyView(view) {
    state = view;
    versionEl.textContent = `v${view.version}`;
    undoBtn.disabled = busy || !view.can_undo;
    redoBtn.disabled = busy || !view.can_redo;
    if (window.showComic) window.showComic(view);
  }

  async function errText(res) {
    try { const j = await res.json(); return j.detail || JSON.stringify(j); } catch { return `${res.status} ${res.statusText}`; }
  }

  async function attach(id) {
    try {
      const res = await fetch(`/api/comics/${id}`);
      if (!res.ok) throw new Error("gone");
      const view = await res.json();
      comicId = id;
      remember(id);
      pending = [];
      renderPending();
      renderHistory(view.chat);
      renderSuggestions();
      dock.classList.remove("hidden");
      applyView(view);
      if (view.busy) { setBusy(true); waitUntilIdle(); }
      return true;
    } catch {
      reset(true);
      return false;
    }
  }

  async function waitUntilIdle() { // page was reloaded while an edit was running
    addNote("An edit is still being applied…");
    const tick = async () => {
      const view = await (await fetch(`/api/comics/${comicId}`)).json();
      if (view.busy) return setTimeout(tick, 2500);
      renderHistory(view.chat);
      applyView(view);
      setBusy(false);
    };
    tick();
  }

  function reset(forget = false) {
    comicId = null;
    if (forget) remember(null);
    dock.classList.add("hidden");
    panel.classList.add("hidden");
    dock.classList.remove("open");
    log.textContent = "";
    pending = [];
    renderPending();
  }

  // ------------------------------------------------------------------ open / close
  function open() {
    panel.classList.remove("hidden");
    dock.classList.add("open");
    toggle.setAttribute("aria-expanded", "true");
    input.focus();
    scrollDown();
  }
  function close() {
    panel.classList.add("hidden");
    dock.classList.remove("open");
    toggle.setAttribute("aria-expanded", "false");
    toggle.focus();
  }
  toggle.addEventListener("click", open);
  $("chatClose").addEventListener("click", close);
  panel.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });

  // ------------------------------------------------------------------ attachments
  function addFiles(list) {
    for (const f of list) {
      const ext = "." + (f.name.split(".").pop() || "").toLowerCase();
      if (!ALLOWED.includes(ext)) { addNote(`"${f.name}" isn't supported. Attach PDF, DOCX, PNG, JPG or SVG.`); continue; }
      if (f.size > MAX_MB * 1024 * 1024) { addNote(`"${f.name}" is over ${MAX_MB} MB.`); continue; }
      if (pending.length >= MAX_FILES) { addNote(`You can attach up to ${MAX_FILES} files per message.`); break; }
      pending.push(f);
    }
    renderPending();
  }

  $("chatAttach").addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => { addFiles(fileInput.files); fileInput.value = ""; });
  input.addEventListener("paste", (e) => {
    const files = [...(e.clipboardData?.files || [])];
    if (files.length) { e.preventDefault(); addFiles(files); }
  });
  ["dragenter", "dragover"].forEach((t) => panel.addEventListener(t, (e) => { e.preventDefault(); if (!busy) panel.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((t) => panel.addEventListener(t, (e) => { e.preventDefault(); panel.classList.remove("dragging"); }));
  panel.addEventListener("drop", (e) => { if (!busy) addFiles(e.dataTransfer.files); });

  // ------------------------------------------------------------------ sending
  async function followJob(jobId) {
    const bar = addProgress();
    return new Promise((resolve) => {
      const poll = async () => {
        try {
          const job = await (await fetch(`/api/jobs/${jobId}`)).json();
          bar.set(job);
          if (job.state === "error") {
            bar.el.remove();
            addMsg("assistant", `That edit didn't work: ${job.error}\nYour comic is still on its last good version.`, { kind: "error" });
            return resolve();
          }
          if (job.state === "done") {
            bar.el.remove();
            addMsg("assistant", job.summary || "Done.", { kind: "result" });
            const view = await (await fetch(`/api/comics/${comicId}`)).json();
            applyView(view);
            return resolve();
          }
        } catch { /* transient network error: keep polling */ }
        setTimeout(poll, 1200);
      };
      poll();
    });
  }

  async function send() {
    const text = input.value.trim();
    if (busy || !comicId || (!text && !pending.length)) return;
    const files = pending;
    const model = $("modelSelect").value;
    setBusy(true);
    suggestionsEl.textContent = "";
    addMsg("user", text, { atts: files.map((f) => ({ name: f.name, file: f })) });

    const body = new FormData();
    body.append("message", text);
    body.append("model", model);
    files.forEach((f) => body.append("files", f));
    input.value = "";
    pending = [];
    renderPending();
    autosize();

    const typing = addTyping();
    try {
      const res = await fetch(`/api/comics/${comicId}/chat`, { method: "POST", body });
      typing.remove();
      if (!res.ok) throw new Error(await errText(res));
      const data = await res.json();
      addMsg("assistant", data.reply);
      (data.warnings || []).forEach(addNote);
      if (data.job_id) await followJob(data.job_id);
    } catch (err) {
      typing.remove();
      addMsg("assistant", err.message, { kind: "error" });
    }
    setBusy(false);
  }

  form.addEventListener("submit", (e) => { e.preventDefault(); send(); });
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
  });

  async function step(kind) {
    if (busy || !comicId) return;
    setBusy(true);
    try {
      const res = await fetch(`/api/comics/${comicId}/${kind}`, { method: "POST" });
      if (!res.ok) throw new Error(await errText(res));
      const view = await res.json();
      if (view.note) addMsg("assistant", view.note, { kind: "result" });
      applyView(view);
    } catch (err) {
      addMsg("assistant", err.message, { kind: "error" });
    }
    setBusy(false);
  }
  undoBtn.addEventListener("click", () => step("undo"));
  redoBtn.addEventListener("click", () => step("redo"));

  // ------------------------------------------------------------------ public API
  async function refresh() { // the page changed outside the chat (text-size buttons): re-read history + undo state
    if (!comicId) return;
    try { const v = await (await fetch(`/api/comics/${comicId}`)).json(); renderHistory(v.chat); state = v; versionEl.textContent = `v${v.version}`; undoBtn.disabled = busy || !v.can_undo; redoBtn.disabled = busy || !v.can_redo; } catch {}
  }

  window.ComicChat = { attach, reset, refresh, restore: () => { const id = recall(); return id ? attach(id) : Promise.resolve(false); } };
})();

// Reopen the last comic after a page reload (its state lives on the server).
window.ComicChat.restore();
