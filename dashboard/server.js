#!/usr/bin/env node
/* Progress dashboard server — reads the project's log/output files directly.
 * No auth, read-only, local network. Serves dashboard/public on /.
 * Run: node dashboard/server.js   (PORT env, default 8787)
 */
const express = require("express");
const fs = require("fs");
const path = require("path");
const { execFile } = require("child_process");

const ROOT = path.resolve(__dirname, "..");
const LOGS = path.join(ROOT, "logs");
const EVAL = path.join(ROOT, "outputs", "eval");
const PROGRESS = path.join(ROOT, "outputs", "progress");
const CORPUS = path.join(ROOT, "outputs", "corpus");
const PORT = process.env.PORT || 8787;
const DEV = "muse-glimmer-long-ctx-dev-1";

const app = express();
app.use(express.static(path.join(__dirname, "public")));

const readJSONL = (p) => {
  try {
    return fs.readFileSync(p, "utf8").split("\n").filter(Boolean).map((l) => {
      try { return JSON.parse(l); } catch { return null; }
    }).filter(Boolean);
  } catch { return []; }
};
const tailFile = (p, n = 6) => {
  try { return fs.readFileSync(p, "utf8").split("\n").filter(Boolean).slice(-n); }
  catch { return null; }
};
const mtime = (p) => { try { return fs.statSync(p).mtimeMs; } catch { return 0; } };

// ---- grid summary: mean score per (task, ctx) per file ------------------------
function grid(files) {
  const agg = {};
  for (const f of files) {
    for (const r of readJSONL(path.join(EVAL, f))) {
      if (r.error || r.score == null) continue;
      const k = `${r.task}|${r.target_ctx}`;
      (agg[k] = agg[k] || { task: r.task, ctx: r.target_ctx, n: 0, sum: 0, label: r.config_label })
        .n++, agg[k].sum += r.score;
    }
  }
  return Object.values(agg)
    .map((c) => ({ task: c.task, ctx: c.ctx, label: c.label, n: c.n,
                   mean: +(c.sum / c.n).toFixed(3) }))
    .sort((a, b) => a.task.localeCompare(b.task) || a.ctx - b.ctx);
}

// ---- lane/watcher status -------------------------------------------------------
const WATCHERS = ["overnight", "suite_queue", "stage3", "stage4", "stage5",
  "stage6", "stage7", "stage8", "stage9", "suite_lane"];
const MARKERS = [
  "overnight-queue.done", "suite-lane.done", "suite-queue.done", "stage3-queue.done",
  "stage4-queue.done", "stage5-queue.done", "train1.launched", "stage7-queue.done",
  "stage8-queue.done", "stage9-queue.done",
];

function probe(cmd, args) {
  return new Promise((res) => execFile(cmd, args, { timeout: 8000 },
    (e, stdout) => res({ ok: !e, out: String(stdout || e || "").trim() })));
}

const GRIDS = [
  { name: "gt128k (>128k grid)", file: "stock_vllm_gt128k.jsonl", total: 216 },
  { name: "le128k (§3 baseline)", file: "stock_vllm_le128k.jsonl", total: 378 },
  { name: "cwe", file: "stock_cwe.jsonl", total: 36 },
  { name: "NoLiMa suite", file: "suite_nolima.jsonl", total: 54 },
  { name: "LongBench v2", file: "suite_longbench_v2.jsonl", total: 12 },   // 4 ctx × depth 0.5 × 3 reps (suite_lane/suite_queue spec)
  { name: "LongCodeQA", file: "suite_longcodeqa.jsonl", total: 15 },       // 5 ctx × depth 0.5 × 3 reps
  { name: "InfBench", file: "suite_infbench.jsonl", total: 18 },           // 3 tasks × 2 ctx × depth 0.5 × 3 reps
  { name: "synth3 fill-in", file: "suite_synth3.jsonl", total: 189 },
  { name: "agentmem", file: "suite_agentmem.jsonl", total: 72 },         // 6 ctx × 4 depths × 3 reps (stage3 spec)
  { name: "run1 (§8 trained)", file: "run1_vllm.jsonl", total: 162 },     // 6 tasks × 3 ctx × 3 depths × 3 reps (stage7; short file separate)
];

/* ETA from per-row timestamps (each JSONL row carries `ts`, naive UTC from the
 * container). Parse as UTC explicitly (append Z) or JS treats it as local time.
 * Rate = rows completed in the recent window; remaining / rate. Grids share the GPU,
 * so ETAs are upper-ish bounds that tighten as lanes drain. */
const parseTs = (s) => {
  if (!s) return 0;
  return Date.parse(/[Zz]$|[+-]\d\d:?\d\d$/.test(s) ? s : s + "Z");
};
function etaFor(file, rows, total, now, runnerAlive) {
  if (rows.length === 0 || rows.length >= total) return null;
  const ts = rows.map((r) => parseTs(r.ts)).filter(Boolean).sort((a, b) => a - b);
  if (ts.length < 2) return { eta: "…", rate: 0 };
  const W = 60 * 60 * 1000;                       // recent window: 60 min
  const t0 = Math.max(ts[0], now - W), t1 = ts[ts.length - 1];
  const inWin = ts.filter((t) => t >= t0).length;
  if (inWin >= 2 && t1 > t0) {
    const rate = inWin / ((Math.min(t1, now) - t0) / 1000);
    const remain = (total - rows.length) / rate;
    const h = Math.floor(remain / 3600), m = Math.round((remain % 3600) / 60);
    return { eta: (h ? h + "h" : "") + m + "m", rate,
             done: new Date(now + remain * 1000) };
  }
  // too few recent completions to estimate a rate (long cells are normal at 128k+:
  // single cells legitimately run 45–90 min while lanes share the GPU)
  const age = now - t1;
  if (age > 45 * 60 * 1000)
    return runnerAlive ? { eta: "in flight", rate: 0 }
                       : { eta: "stalled?", rate: 0 };
  return { eta: "…", rate: 0 };
}

/* Stage chain for the pipeline panel: progress files are authoritative; stages
 * without one are synthesized from markers (done) — e.g. stages that completed
 * before per-stage reporting existed — or from live grid rows (stage3's two
 * substages: agentmem grid + ppl probe). */
const STAGE_ORDER = [
  "overnight", "suite", "suite_lane", "stage3", "stage4", "stage5",
  "stage6", "stage7", "stage8", "stage9",
];
const STAGE_MARKER = {
  overnight: "overnight-queue.done", suite: "suite-queue.done",
  suite_lane: "suite-lane.done", stage3: "stage3-queue.done",
  stage4: "stage4-queue.done", stage5: "stage5-queue.done",
  stage6: "train1.launched", stage7: "stage7-queue.done",
  stage8: "stage8-queue.done", stage9: "stage9-queue.done",
};
function readStages(now, runnersAlive) {
  const out = {};
  for (const name of STAGE_ORDER) {
    const f = path.join(PROGRESS, `${name}.json`);
    let doc = null;
    try { doc = JSON.parse(fs.readFileSync(f, "utf8")); } catch { /* none */ }
    if (doc) { out[name] = doc; continue; }
    const marker = STAGE_MARKER[name];
    const mpath = marker && path.join(LOGS, marker);
    if (mpath && fs.existsSync(mpath)) {
      out[name] = { stage: name, state: fs.readFileSync(mpath, "utf8").startsWith("blocked") ? "blocked" : "done",
                   detail: fs.readFileSync(mpath, "utf8").trim().slice(0, 90),
                   done: 1, total: 1, updated: fs.statSync(mpath).mtimeMs / 1000 };
    }
  }
  // stage3 synthesis (pre-reporting script still mid-flight): agentmem + ppl rows
  if (!out.stage3) {
    const am = readJSONL(path.join(EVAL, "suite_agentmem.jsonl")).length;
    const ppl = readJSONL(path.join(EVAL, "ppl_stock.jsonl")).length;
    const total = 72 + 10, done = am + ppl;
    out.stage3 = { stage: "stage3", state: done >= total ? "done" : "running",
      detail: `agentmem ${am}/72 · ppl ${ppl}/10 (synthesized from grid rows)`,
      done, total, updated: now / 1000,
      eta_human: etaFor("suite_agentmem.jsonl", readJSONL(path.join(EVAL, "suite_agentmem.jsonl")), 72, now, runnersAlive["suite_agentmem.jsonl"])?.eta || null };
  }
  return STAGE_ORDER.map((k) => out[k]).filter(Boolean);
}

app.get("/api/status", async (req, res) => {
  const now = Date.now();
  // runner-aliveness per grid (by output filename in the process cmdline)
  const alive = {};
  for (const g of GRIDS) {
    if (!fs.existsSync(path.join(EVAL, g.file))) continue;
    const r = await probe("docker", ["exec", DEV, "pgrep", "-f", `run_eval.py.*${g.file}`]);
    alive[g.file] = r.ok && r.out.length > 0;
  }
  const stages = readStages(now, alive);
  const grids = GRIDS.map((g) => {
    const p = path.join(EVAL, g.file);
    const rows = readJSONL(p);
    return { ...g, rows: rows.length, exists: fs.existsSync(p), mtime: mtime(p),
             runner: alive[g.file] || false,
             ...(etaFor(g.file, rows, g.total, now, alive[g.file]) || {}) };
  });
  const watchers = {};
  for (const w of WATCHERS) {
    const r = await probe("pgrep", ["-f", `[s]cripts/${w}`]);
    watchers[w] = r.out.length > 0;
  }
  const markers = Object.fromEntries(MARKERS.map((m) => [
    m, fs.existsSync(path.join(LOGS, m)) ?
      fs.readFileSync(path.join(LOGS, m), "utf8").trim() : null]));
  // log tails per STAGE: {stage: {log, lines}} — logs mapped to their owning stage;
  // multiple logs per stage allowed (e.g. stage4's per-arm grid logs). Longer tails
  // (24 lines) since they now live behind an accordion.
  const STAGE_LOGS = {
    overnight: ["eval-stock-gt128k", "overnight-queue"],
    suite: ["suite-grid-suite_nolima", "suite-grid-suite_longcodeqa",
            "suite-grid-suite_infbench", "suite-grid-suite_synth3", "suite-queue"],
    suite_lane: ["suite-lane"],
    stage3: ["suite-grid-suite_agentmem", "ppl-probe", "stage3-queue"],
    stage4: ["stage4-qk4.3-grid", "stage4-qk5.0-grid", "stage4-yarn4-grid", "stage4-queue"],
    stage5: ["stage5-queue"],
    stage6: ["stage6-queue", "train-run1"],
    stage7: ["stage7-grid", "stage7-queue"],
    stage8: ["export-run1", "stage8-queue"],
    stage9: ["stage9-grid", "stage9-dflash", "stage9-queue"],
  };
  const stageLogs = {};
  for (const [stage, names] of Object.entries(STAGE_LOGS)) {
    for (const n of names) {
      const lines = tailFile(path.join(LOGS, `${n}.log`), 24);
      if (lines && lines.length) {
        (stageLogs[stage] = stageLogs[stage] || []).push({ log: `${n}.log`, lines });
      }
    }
  }
  let corpus = null;
  try {
    corpus = JSON.parse(fs.readFileSync(path.join(CORPUS, "train_v1", "manifest.json"), "utf8"));
  } catch { /* not yet */ }
  const dockerPs = await probe("docker", ["ps", "--format", "{{.Names}} {{.Status}}"]);
  const vllm = await probe("docker", ["exec", DEV, "curl", "-s", "--max-time", "3",
    "http://vllm:8000/v1/models"]);
  res.json({
    ts: new Date().toISOString(),
    stages, grids, watchers, markers, stageLogs, corpus,
    docker: { ps: dockerPs.out, vllm_up: vllm.ok && vllm.out.includes("muse-glimmer") },
  });
});

app.get("/api/results", (req, res) => {
  /* One consolidated results table: ALL grids, one row per (task, ctx), pooled across
   * files with per-source provenance. Duplicate cells (same task+ctx in two files)
   * are pooled by label — arm/run grids keep their own label rows. */
  const agg = {};
  for (const g of GRIDS) {
    for (const r of readJSONL(path.join(EVAL, g.file))) {
      if (r.error || r.score == null) continue;
      const k = `${r.task}|${r.target_ctx}|${r.config_label}`;
      (agg[k] = agg[k] || { task: r.task, ctx: r.target_ctx, label: r.config_label,
                            src: g.name, n: 0, sum: 0 }).n++;
      agg[k].sum += r.score;
    }
  }
  res.json(Object.values(agg)
    .map((c) => ({ task: c.task, ctx: c.ctx, label: c.label, src: c.src, n: c.n,
                   mean: +(c.sum / c.n).toFixed(3) }))
    .sort((a, b) => a.task.localeCompare(b.task) || a.label.localeCompare(b.label) || a.ctx - b.ctx));
});

app.get("/api/grid/:file", (req, res) => {
  const allowed = GRIDS.some((g) => g.file === req.params.file);
  if (!allowed) return res.status(400).json({ error: "unknown grid" });
  res.json(grid([req.params.file]));
});

app.listen(PORT, "0.0.0.0", () =>
  console.log(`dashboard: http://<host>:${PORT}  (root: ${ROOT})`));
