"use strict";

const STORAGE_KEY = "dmt-topics-v1";
const DATA_URL = "data/topics.json";

const STATUS_LABEL = {
  candidate: "후보",
  review: "검토중",
  adopted: "채택",
  rejected: "폐기",
};

const CATEGORY_LABEL = {
  classification: "분류",
  regression: "회귀",
  unsupervised: "비지도",
};

const state = {
  topics: [],
  filter: { status: "all", category: "all", query: "" },
  editingId: null,
};

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

async function init() {
  await loadTopics();
  bindUI();
  render();
}

async function loadTopics() {
  const cached = localStorage.getItem(STORAGE_KEY);
  if (cached) {
    try {
      state.topics = JSON.parse(cached);
      return;
    } catch {
      // fall through to fetch
    }
  }
  try {
    const res = await fetch(DATA_URL, { cache: "no-store" });
    if (!res.ok) throw new Error("fetch failed");
    const data = await res.json();
    state.topics = Array.isArray(data) ? data : data.topics || [];
    persist();
  } catch {
    state.topics = [];
  }
}

function persist() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state.topics));
}

function bindUI() {
  $("#btn-add").addEventListener("click", () => openEditor(null));
  $("#btn-export").addEventListener("click", exportJson);
  $("#file-import").addEventListener("change", importJson);
  $("#btn-reset").addEventListener("click", resetFromFile);
  $("#search").addEventListener("input", (e) => {
    state.filter.query = e.target.value.trim().toLowerCase();
    render();
  });

  bindChipGroup("#filter-status", "status");
  bindChipGroup("#filter-category", "category");

  const dialog = $("#editor");
  $$("[data-close]", dialog).forEach((b) =>
    b.addEventListener("click", () => dialog.close()),
  );
  $("#editor-form").addEventListener("submit", onSave);
  $("[data-delete]", dialog).addEventListener("click", onDelete);
}

function bindChipGroup(selector, key) {
  const group = $(selector);
  group.addEventListener("click", (e) => {
    const btn = e.target.closest(".chip");
    if (!btn) return;
    $$(".chip", group).forEach((c) => c.classList.remove("active"));
    btn.classList.add("active");
    state.filter[key] = btn.dataset.value;
    render();
  });
}

function filteredTopics() {
  const { status, category, query } = state.filter;
  return state.topics.filter((t) => {
    if (status !== "all" && t.status !== status) return false;
    if (category !== "all" && t.category !== category) return false;
    if (query) {
      const hay = [
        t.title,
        t.problem,
        t.dataset,
        (t.models || []).join(" "),
        t.owner,
        t.notes,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      if (!hay.includes(query)) return false;
    }
    return true;
  });
}

function render() {
  const list = filteredTopics();
  renderStats();
  const grid = $("#grid");
  grid.innerHTML = "";
  $("#empty").hidden = list.length !== 0;

  for (const t of list) {
    grid.appendChild(buildCard(t));
  }
}

function renderStats() {
  const total = state.topics.length;
  const byStatus = state.topics.reduce((acc, t) => {
    acc[t.status] = (acc[t.status] || 0) + 1;
    return acc;
  }, {});
  const html = [
    `<span class="stat">전체 <strong>${total}</strong></span>`,
    `<span class="stat">후보 <strong>${byStatus.candidate || 0}</strong></span>`,
    `<span class="stat">검토중 <strong>${byStatus.review || 0}</strong></span>`,
    `<span class="stat">채택 <strong>${byStatus.adopted || 0}</strong></span>`,
    `<span class="stat">폐기 <strong>${byStatus.rejected || 0}</strong></span>`,
  ].join("");
  $("#stats").innerHTML = html;
}

function buildCard(t) {
  const el = document.createElement("article");
  el.className = "card";
  el.dataset.id = t.id;

  const models = (t.models || [])
    .slice(0, 6)
    .map((m) => `<span class="model-pill">${escapeHtml(m)}</span>`)
    .join("");

  const extra =
    t.models && t.models.length > 6
      ? `<span class="model-pill">+${t.models.length - 6}</span>`
      : "";

  el.innerHTML = `
    <div class="card-head">
      <h3 class="card-title">${escapeHtml(t.title || "제목 없음")}</h3>
      <span class="badge status-${t.status}">${STATUS_LABEL[t.status] || t.status}</span>
    </div>
    <div class="card-meta">
      <span class="tag category-${t.category}">${CATEGORY_LABEL[t.category] || t.category}</span>
      ${t.dataset ? `<span class="tag">${escapeHtml(t.dataset)}</span>` : ""}
    </div>
    <p class="card-problem">${escapeHtml(t.problem || "")}</p>
    <div class="card-models">${models}${extra}</div>
    <div class="card-footer">
      <span>${t.owner ? "담당 " + escapeHtml(t.owner) : ""}</span>
      <span>${t.createdAt || ""}</span>
    </div>
  `;

  el.addEventListener("click", () => openEditor(t.id));
  return el;
}

function openEditor(id) {
  const dialog = $("#editor");
  const form = $("#editor-form");
  form.reset();

  const deleteBtn = $("[data-delete]", dialog);
  const topic = id ? state.topics.find((t) => t.id === id) : null;
  state.editingId = id;

  $("#editor-title").textContent = topic ? "주제 편집" : "새 주제";
  deleteBtn.hidden = !topic;

  if (topic) {
    form.elements.title.value = topic.title || "";
    form.elements.category.value = topic.category || "classification";
    form.elements.status.value = topic.status || "candidate";
    form.elements.problem.value = topic.problem || "";
    form.elements.dataset.value = topic.dataset || "";
    form.elements.models.value = (topic.models || []).join(", ");
    form.elements.pros.value = (topic.pros || []).join("\n");
    form.elements.cons.value = (topic.cons || []).join("\n");
    form.elements.owner.value = topic.owner || "";
    form.elements.notes.value = topic.notes || "";
  } else {
    form.elements.category.value = "classification";
    form.elements.status.value = "candidate";
  }

  dialog.showModal();
}

function onSave(e) {
  e.preventDefault();
  const form = e.currentTarget;
  const fd = new FormData(form);
  const topic = {
    id: state.editingId || `t-${Date.now().toString(36)}`,
    title: (fd.get("title") || "").trim(),
    category: fd.get("category"),
    status: fd.get("status"),
    problem: (fd.get("problem") || "").trim(),
    dataset: (fd.get("dataset") || "").trim(),
    models: splitList(fd.get("models"), ","),
    pros: splitList(fd.get("pros"), "\n"),
    cons: splitList(fd.get("cons"), "\n"),
    owner: (fd.get("owner") || "").trim(),
    notes: (fd.get("notes") || "").trim(),
    createdAt: new Date().toISOString().slice(0, 10),
  };

  if (state.editingId) {
    const idx = state.topics.findIndex((t) => t.id === state.editingId);
    if (idx >= 0) {
      topic.createdAt = state.topics[idx].createdAt || topic.createdAt;
      state.topics[idx] = topic;
    }
  } else {
    state.topics.unshift(topic);
  }

  persist();
  $("#editor").close();
  render();
}

function onDelete() {
  if (!state.editingId) return;
  if (!confirm("이 주제를 삭제할까요?")) return;
  state.topics = state.topics.filter((t) => t.id !== state.editingId);
  persist();
  $("#editor").close();
  render();
}

function exportJson() {
  const blob = new Blob(
    [JSON.stringify({ topics: state.topics }, null, 2)],
    { type: "application/json" },
  );
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const stamp = new Date().toISOString().slice(0, 10);
  a.href = url;
  a.download = `topics-${stamp}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function importJson(e) {
  const file = e.target.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const parsed = JSON.parse(reader.result);
      const arr = Array.isArray(parsed) ? parsed : parsed.topics;
      if (!Array.isArray(arr)) throw new Error("invalid format");
      state.topics = arr;
      persist();
      render();
      alert(`${arr.length}개 주제를 불러왔습니다.`);
    } catch (err) {
      alert("JSON 형식이 올바르지 않습니다.");
    }
    e.target.value = "";
  };
  reader.readAsText(file, "utf-8");
}

async function resetFromFile() {
  if (!confirm("저장된 변경사항을 버리고 data/topics.json 으로 초기화할까요?")) return;
  localStorage.removeItem(STORAGE_KEY);
  await loadTopics();
  render();
}

function splitList(raw, sep) {
  return (raw || "")
    .split(sep)
    .map((s) => s.trim())
    .filter(Boolean);
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

init();
