const $ = (id) => document.getElementById(id);
let cast = [];
let styleCatalogue = [];
let textSize = "medium";
let currentComicId = null;

// Everything user- or model-supplied that goes into innerHTML must pass through here.
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function setSeg(group, value) {
  group.querySelectorAll("button").forEach((b) => {
    const on = b.dataset.size === value;
    b.classList.toggle("on", on);
    b.setAttribute("aria-checked", on ? "true" : "false");
  });
}

async function loadStyles() {
  const sel = $("artStyle");
  try {
    const data = await (await fetch("/api/styles")).json();
    styleCatalogue = data.styles;
    const groups = {};
    styleCatalogue.forEach((st) => (groups[st.group] ||= []).push(st));
    sel.innerHTML =
      Object.entries(groups)
        .map(([g, list]) => `<optgroup label="${esc(g)}">${list.map((st) => `<option value="${esc(st.id)}">${esc(st.label)}</option>`).join("")}</optgroup>`)
        .join("") + `<optgroup label="Other"><option value="__custom">Custom style…</option></optgroup>`;
    sel.value = data.default;
  } catch {
    sel.innerHTML = `<option value="classic_superhero">Classic superhero</option>`;
  }
  updateStyleDesc();
}

function updateStyleDesc() {
  const v = $("artStyle").value;
  $("customStyle").classList.toggle("hidden", v !== "__custom");
  const st = styleCatalogue.find((x) => x.id === v);
  $("styleDesc").textContent = v === "__custom" ? "Your own wording is used as the style for every panel." : st ? st.description : "";
}
$("artStyle").addEventListener("change", updateStyleDesc);

$("textSize").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-size]");
  if (!b) return;
  textSize = b.dataset.size;
  setSeg($("textSize"), textSize);
});

function chosenStyle() {
  const v = $("artStyle").value;
  return v === "__custom" ? $("customStyle").value.trim() || "classic_superhero" : v;
}

async function loadModels() {
  const select = $("modelSelect");
  try {
    const res = await fetch("/api/models");
    const data = await res.json();
    select.innerHTML = data.models.map((m) => `<option value="${esc(m)}">${esc(m)}</option>`).join("");
    if (!data.models.length) select.innerHTML = `<option value="">No local models found</option>`;
  } catch {
    select.innerHTML = `<option value="">Couldn't reach the model server</option>`;
  }
}

async function loadCast() {
  const res = await fetch("/api/characters");
  cast = await res.json();
  renderCast();
}

function renderCast() {
  $("castCount").textContent = cast.length;
  $("castList").innerHTML = cast
    .map(
      (c) => `
    <div class="cast-card" data-id="${esc(c.id)}">
      <img src="${esc((c.image_urls || [])[0] || "")}" alt="" onerror="this.style.visibility='hidden'" />
      <div class="cast-body">
        <strong>${esc(c.name)}</strong>
        <p class="cast-desc">${c.description ? esc(c.description) : '<span class="hint">no description — comic will invent their look</span>'}</p>
        <button class="cast-edit" data-edit="${esc(c.id)}" title="Edit description">✎</button>
      </div>
      <button data-remove="${esc(c.id)}" title="Remove">✕</button>
    </div>`
    )
    .join("") || `<p class="hint">No characters yet — the comic will invent its own cast.</p>`;
}

$("castForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const name = $("castName").value.trim();
  const files = $("castFiles").files;
  const description = $("castDesc").value.trim();
  if (!name || !files.length) return;
  const form = new FormData();
  form.append("name", name);
  form.append("description", description);
  for (const f of files) form.append("files", f);
  const res = await fetch("/api/characters", { method: "POST", body: form });
  if (!res.ok) {
    let msg = "Couldn't add that character.";
    try { msg = (await res.json()).detail || msg; } catch {}
    alert(msg);
    return;
  }
  $("castForm").reset();
  await loadCast();
});

$("castList").addEventListener("click", async (e) => {
  const removeId = e.target.dataset.remove;
  const editId = e.target.dataset.edit;
  if (removeId) {
    await fetch(`/api/characters/${removeId}`, { method: "DELETE" });
    await loadCast();
    return;
  }
  if (editId) {
    const c = cast.find((x) => x.id === editId);
    if (!c) return;
    const next = prompt("Describe this character (this is what the comic is drawn from):", c.description || "");
    if (next === null) return;
    const form = new FormData();
    form.append("description", next || "");
    await fetch(`/api/characters/${editId}`, { method: "PATCH", body: form });
    await loadCast();
  }
});

$("panelCount").addEventListener("input", (e) => {
  $("panelCountLabel").textContent = e.target.value;
});

function setBusy(busy) {
  $("generateBtn").disabled = busy;
  $("generateBtn").textContent = busy ? "Working on it..." : "Generate comic!";
}

let liveShown = 0;
function renderLive(urls) {
  const strip = $("liveStrip");
  if (urls.length < liveShown) { strip.textContent = ""; liveShown = 0; }
  for (; liveShown < urls.length; liveShown++) {
    const img = document.createElement("img");
    img.src = urls[liveShown];
    img.alt = `Panel ${liveShown + 1}`;
    strip.appendChild(img);
  }
}

function showWarnings(list) {
  const ul = $("warnings");
  ul.textContent = "";
  list.forEach((w) => { const li = document.createElement("li"); li.textContent = w; ul.appendChild(li); });
  ul.classList.toggle("hidden", !list.length);
}

async function pollJob(jobId) {
  const poll = async () => {
    const res = await fetch(`/api/jobs/${jobId}`);
    const job = await res.json();
    $("progressFill").style.width = `${Math.round(job.progress * 100)}%`;
    $("progressMessage").textContent = job.message;
    renderLive(job.panel_urls || []);

    if (job.state === "error") {
      setBusy(false);
      $("progressArea").classList.add("hidden");
      $("errorArea").classList.remove("hidden");
      $("errorArea").textContent = `Couldn't finish the comic: ${job.error}`;
      return;
    }
    if (job.state === "done") {
      setBusy(false);
      $("progressArea").classList.add("hidden");
      $("resultArea").classList.remove("hidden");
      $("pageImage").src = job.page_url + `?t=${Date.now()}`;
      $("downloadBtn").href = job.page_url;
      $("downloadBtn").download = "comic.png";
      currentComicId = job.id;
      setSeg($("resultTextSize"), textSize);
      showWarnings(job.warnings || []);
      if (window.ComicChat) window.ComicChat.attach(job.id); // the job id is also the comic id
      return;
    }
    setTimeout(poll, 1200);
  };
  poll();
}

$("generateForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("errorArea").classList.add("hidden");
  $("resultArea").classList.add("hidden");
  if (window.ComicChat) window.ComicChat.reset(true); // a new comic replaces the one the chat was editing
  $("progressArea").classList.remove("hidden");
  $("progressFill").style.width = "0%";
  $("progressMessage").textContent = "Queuing up...";
  $("liveStrip").textContent = "";
  liveShown = 0;
  showWarnings([]);
  setBusy(true);

  const body = {
    prompt: $("prompt").value.trim(),
    model: $("modelSelect").value,
    panel_count: Number($("panelCount").value),
    art_style: chosenStyle(),
    text_size: textSize,
    quality: $("quality").value,
    character_ids: cast.map((c) => c.id),
  };

  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await res.text());
    const { job_id } = await res.json();
    pollJob(job_id);
  } catch (err) {
    setBusy(false);
    $("progressArea").classList.add("hidden");
    $("errorArea").classList.remove("hidden");
    $("errorArea").textContent = `Couldn't start the comic: ${err.message}`;
  }
});

$("againBtn").addEventListener("click", () => {
  $("resultArea").classList.add("hidden");
  $("prompt").focus();
});

// Called by the chat whenever the comic changes (an edit, undo/redo, or restoring after a reload).
$("resultTextSize").addEventListener("click", async (e) => {
  const b = e.target.closest("button[data-size]");
  if (!b || !currentComicId || b.dataset.size === textSize) return;
  const group = $("resultTextSize");
  group.querySelectorAll("button").forEach((x) => (x.disabled = true));
  $("resizeNote").textContent = "Re-lettering…";
  try {
    const res = await fetch(`/api/comics/${currentComicId}/text_size`, { method: "POST", body: new URLSearchParams({ text_size: b.dataset.size }) });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const view = await res.json();
    window.showComic(view);
    if (window.ComicChat) window.ComicChat.refresh();
    $("resizeNote").textContent = "Done. Undo is in the chat.";
  } catch (err) {
    $("resizeNote").textContent = `Couldn't change it: ${err.message}`;
  }
  group.querySelectorAll("button").forEach((x) => (x.disabled = false));
});

window.showComic = (view) => {
  currentComicId = view.id || currentComicId;
  if (view.text_size) { textSize = view.text_size; setSeg($("textSize"), textSize); setSeg($("resultTextSize"), textSize); }
  $("resultArea").classList.remove("hidden");
  $("pageImage").src = view.page_url + `?v=${view.version}`;
  $("downloadBtn").href = view.page_url;
  $("downloadBtn").download = "comic.png";
};

loadModels();
loadStyles();
loadCast();
