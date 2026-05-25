// Gomoku frontend. Vanilla JS, no framework. Same-origin FastAPI backend.

const BLACK = 1;
const WHITE = -1;
const LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"; // skip 'I' per Go convention

const els = {
  board:        document.getElementById("board"),
  blackSel:     document.getElementById("black-select"),
  whiteSel:     document.getElementById("white-select"),
  sizeSel:      document.getElementById("size-select"),
  newBtn:       document.getElementById("new-game-btn"),
  speed:        document.getElementById("speed"),
  speedReadout: document.getElementById("speed-readout"),
  playBtn:      document.getElementById("play-btn"),
  pauseBtn:     document.getElementById("pause-btn"),
  stepBtn:      document.getElementById("step-btn"),
  toMove:       document.getElementById("to-move"),
  plyCount:     document.getElementById("ply-count"),
  gameId:       document.getElementById("game-id"),
  result:       document.getElementById("result-readout"),
  resultVal:    document.getElementById("result-val"),
  sheetBody:    document.getElementById("sheet-body"),
  sheetEmpty:   document.getElementById("sheet-empty"),
  agentSummary: document.getElementById("agent-summary"),
  statusDot:    document.getElementById("status-dot"),
  statusLabel:  document.getElementById("status-label"),
  exportBtn:    document.getElementById("export-btn"),
  footerHost:   document.getElementById("footer-host"),
  tabs:         document.querySelectorAll(".tab"),
  playersBlock: document.getElementById("players-block"),
  watchBlock:   document.getElementById("watch-block"),
  replayBlock:  document.getElementById("replay-block"),
  replayFile:   document.getElementById("replay-file"),
  replayCtrls:  document.getElementById("replay-controls"),
  repBack:      document.getElementById("rep-back"),
  repFwd:       document.getElementById("rep-fwd"),
  repPlay:      document.getElementById("rep-play"),
};

const state = {
  mode: "hva",
  agents: [],
  game: null,
  hover: null,
  watch:  { playing: false, timer: null },
  replay: { plies: null, idx: 0, playing: false, timer: null, size: 15 },
};

// ─── init ─────────────────────────────────────────────────────────────

async function init() {
  els.footerHost.textContent = location.host;
  try {
    state.agents = (await (await fetch("/api/agents")).json()).agents;
  } catch { state.agents = []; }
  populateAgentSelects();
  attachHandlers();
  await newGame();
  renderAll();
}

function populateAgentSelects() {
  for (const sel of [els.blackSel, els.whiteSel]) {
    sel.innerHTML = "";
    const human = document.createElement("option");
    human.value = "human"; human.textContent = "human";
    sel.appendChild(human);
    for (const a of state.agents) {
      const o = document.createElement("option");
      o.value = `agent:${a.name}`;
      o.textContent = a.name;
      o.title = a.doc || a.module;
      sel.appendChild(o);
    }
    if (state.agents.length === 0) {
      const dis = document.createElement("option");
      dis.value = ""; dis.disabled = true;
      dis.textContent = "—  no agents registered  —";
      sel.appendChild(dis);
    }
  }
  els.blackSel.value = "human";
  els.whiteSel.value = state.agents.length > 0 ? `agent:${state.agents[0].name}` : "human";
}

function attachHandlers() {
  els.newBtn.addEventListener("click", newGame);
  els.tabs.forEach(t => t.addEventListener("click", () => switchMode(t.dataset.mode)));
  els.speed.addEventListener("input", () => {
    els.speedReadout.textContent = `${els.speed.value}ms`;
  });
  els.playBtn.addEventListener("click", startWatch);
  els.pauseBtn.addEventListener("click", stopWatch);
  els.stepBtn.addEventListener("click", stepAgentOnce);
  els.exportBtn.addEventListener("click", exportJson);
  els.replayFile.addEventListener("change", loadReplayFile);
  els.repBack.addEventListener("click", () => seekReplay(state.replay.idx - 1));
  els.repFwd.addEventListener("click", () => seekReplay(state.replay.idx + 1));
  els.repPlay.addEventListener("click", toggleReplayPlay);
  els.board.addEventListener("click", onBoardClick);
  els.board.addEventListener("mousemove", onBoardMove);
  els.board.addEventListener("mouseleave", () => { state.hover = null; renderBoard(); });
}

// ─── modes ────────────────────────────────────────────────────────────

function switchMode(m) {
  state.mode = m;
  els.tabs.forEach(t => t.setAttribute("aria-selected", t.dataset.mode === m ? "true" : "false"));
  els.watchBlock.hidden  = (m !== "ava");
  els.replayBlock.hidden = (m !== "rep");
  els.playersBlock.hidden = (m === "rep");
  stopWatch();
  stopReplayPlay();
  if (m === "rep") {
    state.replay.plies = null;
    state.replay.idx = 0;
    els.replayCtrls.hidden = true;
    state.game = null;
    renderAll();
    setStatus("idle", "load a json");
  } else {
    if (m === "ava" && state.agents.length > 0) {
      els.blackSel.value = `agent:${state.agents[0].name}`;
      els.whiteSel.value = `agent:${state.agents[state.agents.length > 1 ? 1 : 0].name}`;
    } else if (m === "hva") {
      els.blackSel.value = "human";
      els.whiteSel.value = state.agents.length > 0 ? `agent:${state.agents[0].name}` : "human";
    }
    newGame();
  }
}

// ─── game flow ────────────────────────────────────────────────────────

async function newGame() {
  if (state.mode === "rep") return;
  stopWatch();
  const body = {
    size:  parseInt(els.sizeSel.value, 10),
    black: els.blackSel.value || "human",
    white: els.whiteSel.value || "human",
  };
  if (!body.black || !body.white) { setStatus("idle", "pick players"); return; }
  setStatus("live", "new game");
  try {
    const r = await fetch("/api/games", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      setStatus("idle", `error: ${err.detail || r.statusText}`);
      return;
    }
    state.game = await r.json();
    renderAll();
    if (state.game.next_actor === "agent" && state.mode === "hva") {
      await stepAgentOnce();
    }
  } catch { setStatus("idle", "network error"); }
}

async function stepAgentOnce() {
  if (!state.game || state.game.next_actor !== "agent") return;
  setStatus("thinking", "agent thinking");
  try {
    const r = await fetch(`/api/games/${state.game.game_id}/agent`, { method: "POST" });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      setStatus("idle", `agent error: ${err.detail || r.statusText}`);
      return;
    }
    state.game = await r.json();
    renderAll();
    if (!state.game.terminal && state.mode === "hva" && state.game.next_actor === "agent") {
      await stepAgentOnce();
    }
  } catch { setStatus("idle", "network error"); }
}

async function humanMove(action) {
  if (!state.game || state.game.next_actor !== "human") return;
  const sz = state.game.size;
  if (state.game.board[Math.floor(action / sz)][action % sz] !== 0) return;
  try {
    const r = await fetch(`/api/games/${state.game.game_id}/move`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({action}),
    });
    if (!r.ok) return;
    state.game = await r.json();
    renderAll();
    if (state.game.next_actor === "agent") await stepAgentOnce();
  } catch {}
}

function startWatch() {
  if (!state.game || state.game.terminal || state.mode !== "ava") return;
  state.watch.playing = true;
  els.playBtn.disabled  = true;
  els.pauseBtn.disabled = false;
  const tick = async () => {
    if (!state.watch.playing) return;
    if (!state.game || state.game.terminal || state.game.next_actor !== "agent") {
      stopWatch(); return;
    }
    await stepAgentOnce();
    if (state.watch.playing && state.game && !state.game.terminal) {
      state.watch.timer = setTimeout(tick, parseInt(els.speed.value, 10));
    } else {
      stopWatch();
    }
  };
  tick();
}

function stopWatch() {
  state.watch.playing = false;
  if (state.watch.timer) { clearTimeout(state.watch.timer); state.watch.timer = null; }
  els.playBtn.disabled  = false;
  els.pauseBtn.disabled = true;
}

// ─── replay ───────────────────────────────────────────────────────────

async function loadReplayFile(e) {
  const file = e.target.files?.[0];
  if (!file) return;
  let json;
  try { json = JSON.parse(await file.text()); }
  catch { setStatus("idle", "invalid json"); return; }
  const size  = json.size ?? 15;
  const moves = json.moves;
  if (!Array.isArray(moves)) { setStatus("idle", "json needs {size, moves}"); return; }
  setStatus("live", "validating replay");
  try {
    const r = await fetch("/api/replay", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({size, moves}),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      setStatus("idle", `replay error: ${err.detail || r.statusText}`);
      return;
    }
    const data = await r.json();
    state.replay.plies = data.plies;
    state.replay.size  = data.size;
    state.replay.idx   = 0;
    els.replayCtrls.hidden = false;
    setStatus("live", `replay loaded (${data.plies.length - 1} plies)`);
    seekReplay(0);
  } catch { setStatus("idle", "network error"); }
}

function seekReplay(idx) {
  if (!state.replay.plies) return;
  idx = Math.max(0, Math.min(state.replay.plies.length - 1, idx));
  state.replay.idx = idx;
  const cur = state.replay.plies[idx];
  state.game = {
    game_id: `replay/${idx}`,
    size: state.replay.size,
    board: cur.board,
    to_play: cur.to_play,
    terminal: cur.terminal,
    winner: cur.winner,
    last_move: cur.action,
    history: state.replay.plies.slice(1, idx + 1).map(p => ({
      ply: p.ply, player: p.player, action: p.action, ms: 0, debug: null,
    })),
    black: "replay",
    white: "replay",
    next_actor: cur.terminal ? "none" : "replay",
  };
  renderAll();
}

function toggleReplayPlay() {
  if (state.replay.playing) { stopReplayPlay(); return; }
  if (!state.replay.plies) return;
  state.replay.playing = true;
  els.repPlay.textContent = "❚❚ pause";
  const step = () => {
    if (!state.replay.playing) return;
    if (state.replay.idx >= state.replay.plies.length - 1) { stopReplayPlay(); return; }
    seekReplay(state.replay.idx + 1);
    state.replay.timer = setTimeout(step, parseInt(els.speed.value, 10));
  };
  step();
}

function stopReplayPlay() {
  state.replay.playing = false;
  if (state.replay.timer) { clearTimeout(state.replay.timer); state.replay.timer = null; }
  if (els.repPlay) els.repPlay.textContent = "▷ play";
}

// ─── export ───────────────────────────────────────────────────────────

function exportJson() {
  if (!state.game) return;
  const data = {
    size: state.game.size,
    moves: state.game.history.map(h => h.action),
    black: state.game.black,
    white: state.game.white,
    winner: state.game.winner,
  };
  const blob = new Blob([JSON.stringify(data, null, 2)], {type: "application/json"});
  const url  = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `gomoku-${state.game.game_id || "game"}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// ─── board rendering (SVG) ────────────────────────────────────────────

const BOARD_VB  = 640;
const BOARD_PAD = 36;
const SVG_NS    = "http://www.w3.org/2000/svg";

const boardCell    = (sz) => (BOARD_VB - 2 * BOARD_PAD) / (sz - 1);
const intersection = (sz, r, c) => {
  const cell = boardCell(sz);
  return { x: BOARD_PAD + c * cell, y: BOARD_PAD + r * cell };
};
const moveNotation = (sz, a) => `${LETTERS[a % sz]}${sz - Math.floor(a / sz)}`;
const makeEmptyBoard = (sz) => Array.from({length: sz}, () => Array(sz).fill(0));

function svgEl(name, attrs = {}) {
  const e = document.createElementNS(SVG_NS, name);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}

function renderBoard() {
  const size  = state.game?.size ?? 15;
  const cell  = boardCell(size);
  const board = state.game?.board ?? makeEmptyBoard(size);
  const last  = state.game?.last_move ?? null;
  const interactive = (state.game?.next_actor === "human");
  const svg = els.board;
  svg.innerHTML = "";

  // Grid lines
  for (let i = 0; i < size; i++) {
    const p = BOARD_PAD + i * cell;
    const edge = i === 0 || i === size - 1;
    svg.appendChild(svgEl("line", {
      x1: p, y1: BOARD_PAD, x2: p, y2: BOARD_VB - BOARD_PAD,
      class: edge ? "grid-edge" : "grid-line",
    }));
    svg.appendChild(svgEl("line", {
      x1: BOARD_PAD, y1: p, x2: BOARD_VB - BOARD_PAD, y2: p,
      class: edge ? "grid-edge" : "grid-line",
    }));
  }

  // Star points
  for (const [r, c] of hoshiPoints(size)) {
    const {x, y} = intersection(size, r, c);
    svg.appendChild(svgEl("circle", {cx: x, cy: y, r: 2.6, class: "hoshi"}));
  }

  // Coordinates
  for (let i = 0; i < size; i++) {
    const p = BOARD_PAD + i * cell;
    const col = svgEl("text", {
      x: p, y: BOARD_VB - BOARD_PAD + 18, "text-anchor": "middle", class: "coord",
    });
    col.textContent = LETTERS[i];
    svg.appendChild(col);
    const row = svgEl("text", {
      x: BOARD_PAD - 12, y: p + 3, "text-anchor": "end", class: "coord",
    });
    row.textContent = `${size - i}`;
    svg.appendChild(row);
  }

  // Stones
  const sr = cell * 0.46;
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const v = board[r][c];
      if (v === 0) continue;
      const {x, y} = intersection(size, r, c);
      const isBlack = v === BLACK;
      svg.appendChild(svgEl("circle", {cx: x + 0.5, cy: y + 1.5, r: sr, class: "stone-shadow"}));
      svg.appendChild(svgEl("circle", {cx: x, cy: y, r: sr,
        class: isBlack ? "stone-black" : "stone-white"}));
      svg.appendChild(svgEl("circle", {
        cx: x - sr * 0.32, cy: y - sr * 0.32, r: sr * 0.4,
        class: isBlack ? "stone-black-hl" : "stone-white-hl",
        opacity: isBlack ? "0.25" : "0.5",
      }));
    }
  }

  // Last-move marker
  if (last !== null && last !== undefined) {
    const r = Math.floor(last / size), c = last % size;
    const {x, y} = intersection(size, r, c);
    svg.appendChild(svgEl("circle", {cx: x, cy: y, r: sr * 0.42, class: "last-marker"}));
  }

  // Hover preview
  if (interactive && state.hover) {
    const {r, c} = state.hover;
    if (board[r] && board[r][c] === 0) {
      const {x, y} = intersection(size, r, c);
      const isBlack = state.game.to_play === BLACK;
      svg.appendChild(svgEl("circle", {
        cx: x, cy: y, r: sr,
        class: `ghost-stone ${isBlack ? "stone-black" : "stone-white"}`,
      }));
    }
  }
}

function hoshiPoints(size) {
  if (size === 19) return [[3,3],[3,9],[3,15],[9,3],[9,9],[9,15],[15,3],[15,9],[15,15]];
  if (size === 15) return [[3,3],[3,11],[7,7],[11,3],[11,11]];
  if (size === 13) return [[3,3],[3,9],[6,6],[9,3],[9,9]];
  if (size === 9)  return [[2,2],[2,6],[4,4],[6,2],[6,6]];
  return [];
}

function svgPoint(e) {
  const pt = els.board.createSVGPoint();
  pt.x = e.clientX; pt.y = e.clientY;
  return pt.matrixTransform(els.board.getScreenCTM().inverse());
}

function nearestIntersection(e) {
  if (!state.game) return null;
  const pt = svgPoint(e);
  const sz = state.game.size;
  const cell = boardCell(sz);
  const r = Math.round((pt.y - BOARD_PAD) / cell);
  const c = Math.round((pt.x - BOARD_PAD) / cell);
  if (r < 0 || r >= sz || c < 0 || c >= sz) return null;
  return {r, c};
}

function onBoardMove(e) {
  const hit = nearestIntersection(e);
  if (!hit) { if (state.hover) { state.hover = null; renderBoard(); } return; }
  if (!state.hover || state.hover.r !== hit.r || state.hover.c !== hit.c) {
    state.hover = hit;
    renderBoard();
  }
}

function onBoardClick(e) {
  if (!state.game || state.game.next_actor !== "human") return;
  const hit = nearestIntersection(e);
  if (!hit) return;
  humanMove(hit.r * state.game.size + hit.c);
}

// ─── side panels ──────────────────────────────────────────────────────

function renderScoresheet() {
  const g = state.game;
  if (!g || g.history.length === 0) {
    els.sheetBody.innerHTML = "";
    els.sheetEmpty.style.display = "block";
    renderAgentSummary();
    return;
  }
  els.sheetEmpty.style.display = "none";

  const rows = [];
  for (const h of g.history) {
    if (h.player === BLACK) {
      rows.push({n: rows.length + 1, black: h, white: null});
    } else {
      if (rows.length === 0 || rows[rows.length - 1].white) {
        rows.push({n: rows.length + 1, black: null, white: h});
      } else {
        rows[rows.length - 1].white = h;
      }
    }
  }

  const recentPly = g.history.length;

  els.sheetBody.innerHTML = "";
  for (const row of rows) {
    const tr = document.createElement("tr");
    if ((row.black && row.black.ply === recentPly) || (row.white && row.white.ply === recentPly)) {
      tr.className = "recent";
    }
    const num = document.createElement("td");
    num.className = "num";
    num.textContent = String(row.n).padStart(2, " ");
    tr.appendChild(num);
    tr.appendChild(cellFor(row.black, g.size));
    tr.appendChild(cellFor(row.white, g.size));
    els.sheetBody.appendChild(tr);
  }
  const wrap = els.sheetBody.closest(".sheet-table-wrap");
  if (wrap) wrap.scrollTop = wrap.scrollHeight;

  renderAgentSummary();
}

function cellFor(entry, size) {
  const td = document.createElement("td");
  if (!entry) {
    td.className = "move empty-cell";
    td.textContent = "—";
    return td;
  }
  td.className = "move";
  const not = document.createElement("span");
  not.className = "notation";
  not.textContent = moveNotation(size, entry.action);
  td.appendChild(not);
  if (entry.ms && entry.ms > 0.05) {
    const ms = document.createElement("span");
    ms.className = "ms";
    ms.textContent = entry.ms >= 1000 ? `${(entry.ms / 1000).toFixed(2)}s` : `${entry.ms.toFixed(0)}ms`;
    td.appendChild(ms);
  }
  if (entry.debug) td.title = JSON.stringify(entry.debug);
  return td;
}

function renderAgentSummary() {
  const g = state.game;
  if (!g) { els.agentSummary.innerHTML = ""; return; }
  const rows = [
    playerSummary("Black ●", g.black, g.history.filter(h => h.player === BLACK)),
    playerSummary("White ○", g.white, g.history.filter(h => h.player === WHITE)),
  ];
  els.agentSummary.innerHTML = rows.join("");
}

function playerSummary(label, slot, plies) {
  const slotLabel =
    slot === "human"  ? "human"  :
    slot === "replay" ? "—"       :
    slot.split(":")[1];
  const timed = plies.filter(p => p.ms > 0);
  let avg = "";
  if (timed.length > 0) {
    const m = timed.reduce((s, p) => s + p.ms, 0) / timed.length;
    avg = m >= 1000 ? ` · ⟨${(m/1000).toFixed(2)}s⟩` : ` · ⟨${m.toFixed(0)}ms⟩`;
  }
  return `<div class="row"><span>${label}</span><span class="val">${slotLabel}${avg}</span></div>`;
}

function renderReadouts() {
  const g = state.game;
  if (!g) {
    els.toMove.textContent = "—";
    els.plyCount.textContent = "0";
    els.gameId.textContent = "—";
    els.result.hidden = true;
    return;
  }
  els.plyCount.textContent = g.history.length;
  els.gameId.textContent = String(g.game_id).slice(0, 8);

  if (g.terminal) {
    els.toMove.textContent = "—";
    els.result.hidden = false;
    els.result.classList.remove("lose");
    if (g.winner === 0) {
      els.resultVal.textContent = "draw";
    } else {
      const w = g.winner === BLACK ? "Black ●" : "White ○";
      els.resultVal.textContent = `${w} wins`;
    }
  } else {
    const t = g.to_play === BLACK ? "Black ●" : "White ○";
    const who = g.next_actor === "human" ? "you" : g.next_actor === "agent" ? "agent" : "—";
    els.toMove.textContent = `${t} · ${who}`;
    els.result.hidden = true;
  }
}

function renderAll() {
  renderBoard();
  renderScoresheet();
  renderReadouts();
  updateLiveStatus();
}

// ─── status ───────────────────────────────────────────────────────────

function setStatus(kind, label) {
  els.statusDot.className = `dot ${kind}`;
  els.statusLabel.textContent = label;
}

function updateLiveStatus() {
  const g = state.game;
  if (!g) { setStatus("idle", "idle"); return; }
  if (g.terminal) {
    setStatus(g.winner === 0 ? "idle" : "win", g.winner === 0 ? "draw" : "game over");
    return;
  }
  if (g.next_actor === "human") setStatus("live", "awaiting your move");
  else if (g.next_actor === "agent") setStatus("live", "agent to move");
  else setStatus("idle", "—");
}

init();
