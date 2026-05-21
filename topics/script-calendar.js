"use strict";

const STORAGE_KEY = "dm-calendar-progress";
const DATA_URL = "data/calendar.json";

const STATUS_LABEL = {
  pending: "대기",
  "in-progress": "진행중",
  done: "완료",
};

const STATUS_ICON = {
  pending: "⚪",
  "in-progress": "🟢",
  done: "✅",
};

const state = {
  data: null,
  progress: loadProgress(),
};

const $ = (sel, root = document) => root.querySelector(sel);

init();

async function init() {
  try {
    const res = await fetch(DATA_URL, { cache: "no-store" });
    if (!res.ok) throw new Error("fetch failed");
    state.data = await res.json();
  } catch {
    $("#cal-empty").hidden = false;
    $("#header-subtitle").textContent = "데이터 로드 실패";
    return;
  }

  renderHeader();
  renderMilestones();
  renderToday();
  renderOverall();
}

function loadProgress() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function persistProgress() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state.progress));
}

function isDone(task) {
  if (state.progress[task.id] !== undefined) return !!state.progress[task.id];
  return !!task.done;
}

function setDone(taskId, done) {
  state.progress[taskId] = done;
  persistProgress();
}

function today() {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  return d;
}

function parseDate(s) {
  const d = new Date(s + "T00:00:00");
  d.setHours(0, 0, 0, 0);
  return d;
}

function diffDays(from, to) {
  const ms = to.getTime() - from.getTime();
  return Math.round(ms / 86400000);
}

function fmtRange(start, end) {
  const s = start.slice(5).replace("-", "/");
  const e = end.slice(5).replace("-", "/");
  return `${s} – ${e}`;
}

function fmtShort(iso) {
  return iso.slice(5).replace("-", "/");
}

function renderHeader() {
  const sem = state.data.semester;
  $("#header-subtitle").textContent = sem.project;

  const dleft = diffDays(today(), parseDate(sem.endDate));
  const ddayEl = $("#dday-value");
  if (dleft > 0) {
    ddayEl.textContent = `D-${dleft}`;
  } else if (dleft === 0) {
    ddayEl.textContent = "D-Day";
    $("#dday-pill").classList.add("urgent");
  } else {
    ddayEl.textContent = `+${Math.abs(dleft)}일 경과`;
    $("#dday-pill").classList.add("overdue");
  }
}

function renderMilestones() {
  const root = $("#milestones");
  root.innerHTML = "";
  for (const m of state.data.milestones) {
    root.appendChild(buildMilestoneCard(m));
  }
}

function buildMilestoneCard(m) {
  const article = document.createElement("article");
  const derivedStatus = deriveMilestoneStatus(m);
  article.className = `milestone milestone-${derivedStatus}`;
  article.dataset.id = m.id;

  const tasks = m.tasks || [];
  const total = tasks.length;
  const done = tasks.filter(isDone).length;
  const pct = total ? Math.round((done / total) * 100) : 0;

  article.innerHTML = `
    <header class="milestone-head">
      <div class="milestone-title-row">
        <span class="milestone-icon">${STATUS_ICON[derivedStatus]}</span>
        <h2 class="milestone-title">Week ${m.week}: ${escapeHtml(m.title)}</h2>
        <span class="milestone-status status-${derivedStatus}">${STATUS_LABEL[derivedStatus]}</span>
      </div>
      <div class="milestone-meta">
        <span class="meta-range">📅 ${fmtRange(m.startDate, m.endDate)}</span>
        <span class="meta-progress">${done} / ${total} · ${pct}%</span>
      </div>
      <div class="progress-track milestone-progress">
        <div class="progress-fill" style="width:${pct}%"></div>
      </div>
    </header>
    <ul class="task-list"></ul>
  `;

  const ul = article.querySelector(".task-list");
  for (const t of tasks) {
    ul.appendChild(buildTaskRow(t));
  }

  return article;
}

function deriveMilestoneStatus(m) {
  const tasks = m.tasks || [];
  if (tasks.length === 0) return m.status || "pending";
  const allDone = tasks.every(isDone);
  if (allDone) return "done";
  const someDone = tasks.some(isDone);
  if (someDone) return "in-progress";
  const t0 = today();
  if (parseDate(m.startDate) <= t0 && t0 <= parseDate(m.endDate)) {
    return "in-progress";
  }
  return m.status || "pending";
}

function buildTaskRow(t) {
  const li = document.createElement("li");
  const done = isDone(t);
  const t0 = today();
  const due = parseDate(t.dueDate);
  const dleft = diffDays(t0, due);
  const overdue = !done && dleft < 0;

  li.className = "task-row";
  if (done) li.classList.add("done");
  if (overdue) li.classList.add("overdue");

  const dueLabel = done
    ? ""
    : dleft === 0
      ? `<span class="task-due urgent">오늘 마감</span>`
      : dleft > 0
        ? `<span class="task-due">D-${dleft}</span>`
        : `<span class="task-due overdue">+${Math.abs(dleft)}일 경과</span>`;

  li.innerHTML = `
    <label class="task-check">
      <input type="checkbox" data-task-id="${escapeAttr(t.id)}" ${done ? "checked" : ""} />
      <span class="checkmark" aria-hidden="true"></span>
    </label>
    <div class="task-main">
      <div class="task-title">${escapeHtml(t.title)}</div>
      <div class="task-meta">
        <span class="task-owner">👤 ${escapeHtml(t.owner)}</span>
        <span class="task-date">📅 ${fmtShort(t.dueDate)}</span>
        ${dueLabel}
      </div>
    </div>
  `;

  const input = li.querySelector("input[type=checkbox]");
  input.addEventListener("change", () => {
    setDone(t.id, input.checked);
    renderMilestones();
    renderToday();
    renderOverall();
  });

  return li;
}

function renderToday() {
  const section = $("#today-section");
  const list = $("#today-list");
  list.innerHTML = "";
  const t0 = today();

  const todays = [];
  for (const m of state.data.milestones) {
    for (const t of m.tasks || []) {
      if (isDone(t)) continue;
      const due = parseDate(t.dueDate);
      if (due.getTime() === t0.getTime()) {
        todays.push({ task: t, milestone: m });
      }
    }
  }

  if (todays.length === 0) {
    section.hidden = true;
    return;
  }

  section.hidden = false;
  for (const { task, milestone } of todays) {
    const li = document.createElement("li");
    li.className = "today-item";
    li.innerHTML = `
      <span class="today-pill">W${milestone.week}</span>
      <span class="today-title">${escapeHtml(task.title)}</span>
      <span class="today-owner">${escapeHtml(task.owner)}</span>
    `;
    list.appendChild(li);
  }
}

function renderOverall() {
  let total = 0;
  let done = 0;
  for (const m of state.data.milestones) {
    for (const t of m.tasks || []) {
      total += 1;
      if (isDone(t)) done += 1;
    }
  }
  const pct = total ? Math.round((done / total) * 100) : 0;
  $("#overall-pct").textContent = `${pct}%`;
  $("#overall-bar").style.width = `${pct}%`;
  $("#overall-caption").textContent = `${done} / ${total} 완료`;
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function escapeAttr(str) {
  return escapeHtml(str);
}
