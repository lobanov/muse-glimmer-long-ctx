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
    }).filter(Boolean).filter((r) => !r.error);  // error rows never count as progress
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
  { name: "stock weak5 (enrich)", file: "stock_weak5.jsonl", total: 20 },   // audit F-1.1 enrichment
  { name: "arm qk4.3 (§4)", file: "arm_qk4.3.jsonl", total: 47 },             // 30 primary +5 harm +12 infb (+20 if 512k ext)
  { name: "arm qk5.0 (§4)", file: "arm_qk5.0.jsonl", total: 47 },
  { name: "arm yarn4 (control)", file: "arm_yarn4.jsonl", total: 9 },
  { name: "gt128k (>128k grid)", file: "stock_vllm_gt128k.jsonl", total: 216 },
  { name: "le128k (§3 baseline)", file: "stock_vllm_le128k.jsonl", total: 378 },
  { name: "cwe", file: "stock_cwe.jsonl", total: 36 },
  { name: "NoLiMa suite", file: "suite_nolima.jsonl", total: 54 },
  { name: "LongBench v2", file: "suite_longbench_v2.jsonl", total: 12 },   // 4 ctx × depth 0.5 × 3 reps (suite_lane/suite_queue spec)
  { name: "LongCodeQA", file: "suite_longcodeqa.jsonl", total: 15 },       // 5 ctx × depth 0.5 × 3 reps
  { name: "InfBench", file: "suite_infbench.jsonl", total: 18 },           // 3 tasks × 2 ctx × depth 0.5 × 3 reps
  { name: "RULER 128-512k (stock)", file: "ruler_gt128k.jsonl", total: 48 },
  { name: "MRCR v2 (stock)", file: "mrcr_gt128k.jsonl", total: 12 },
  { name: "synth3 fill-in", file: "suite_synth3.jsonl", total: 189 },
  { name: "agentmem", file: "suite_agentmem.jsonl", total: 72 },         // 6 ctx × 4 depths × 3 reps (stage3 spec)
  { name: "run1 (§8 trained)", file: "run1_vllm.jsonl", total: 177 },     // 6×3×3×3 grid + corroborators: nolima 6, LQA 3, niah_multi 6 (audit F-5.2)
  { name: "run1 short (≤32k regression)", file: "run1_vllm_short.jsonl", total: 18 },
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
      const mtxt = fs.readFileSync(mpath, "utf8");
      // any FAILED anywhere (e.g. "dry-run FAILED :: verify-env") renders non-done
      const mstate = /FAILED/.test(mtxt) ? "failed" :
        (/^blocked/.test(mtxt) ? "blocked" : /^(failed|skipped)/.test(mtxt) ? "failed" : "done");
      out[name] = { stage: name, state: mstate,
                   detail: mtxt.trim().slice(0, 90),
                   done: mstate === "done" ? 1 : 0, total: 1, updated: fs.statSync(mpath).mtimeMs / 1000 };
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
    stage4: ["stage4-stockweak5-grid", "stage4-qk4.3-grid", "stage4-qk5.0-grid", "stage4-yarn4-grid", "stage4-queue"],
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

app.get("/api/workloads", (req, res) => {
  /* Goal afe6584b workload ledger: every eval workload in the current campaign with
   * state = running | queued | complete | partial | invalid, derived from:
   *   markers logs/infbands|*.done, live run_eval argv (docker exec), JSONL rows,
   *   .soup-invalid renames. Queued = a waiting chain script (wait-loop pids). */
  const { execSync } = require("child_process");
  const mk = (m) => fs.existsSync(path.join(LOGS, m));
  let live = [];
  try {
    live = execSync(
      "docker exec " + DEV + " pgrep -af run_eval 2>/dev/null || true",
      { timeout: 8000, encoding: "utf8" }).trim().split("\n")
      .filter((l) => l.includes("run_eval.py")).map((l) => l.trim());
  } catch { /* dev down */ }
  const rowsMatch = (frag) => live.some((l) => l.includes(frag));
  const countRows = (f) => { try {
    return fs.readFileSync(path.join(EVAL, f), "utf8").trim().split("\n")
      .filter((l) => l && !l.includes('"error": "')).length; } catch { return 0; } };

  const W = [];
  const push = (id, group, label, total, marker, file, liveFrag) => {
    const rows = file ? countRows(file) : 0;
    let state = "queued";
    if (mk(marker)) state = rows >= total ? "complete" : "partial";
    else if (rowsMatch(liveFrag || id)) state = "running";
    else if (rows >= total) state = "complete (unmarked)";
    else if (rows > 0) state = "partial";
    W.push({ id, group, label, total, rows, state });
  };

  // ---- goal afe6584b lanes (marker dirs: logs/infbands/ + logs/) ----
  const bands = [
    ["infb_codedebug", "100-140k", 3, 100000, 140000],
    ["infb_codedebug", "140-170k", 3, 140000, 170000],
    ["infb_codedebug", "170-200k", 3, 170000, 200000],
  ];
  for (const [task, tag, n] of bands) {
    const capFile = `infb_${task}_${tag}_capability.jsonl`;
    if (!fs.existsSync(path.join(EVAL, capFile + ".soup-invalid")))
      push(`${task}-${tag}-capability`, "infbench bands", `${task} ${tag} (sampled)`,
           n, `infbands/${task}_${tag}_capability.done`, capFile, `--tasks ${task}`);
    const parFile = `infb_${task}_${tag}_parity.jsonl`;
    if (fs.existsSync(path.join(EVAL, parFile)))  // only ran when sampled was 0.000
      push(`${task}-${tag}-parity`, "infbench bands", `${task} ${tag} (greedy confirm)`,
           n, `infbands/${task}_${tag}_parity.done`, parFile, null);
  }
  const bands2 = [["100-160k"], ["160-220k"], ["220-300k"], ["300-400k"], ["400-510k"]];
  for (const [tag] of bands2) {
    const file = `infb_infb_bookmc_${tag}_capability.jsonl`;
    push(`bookmc-${tag}`, "infbench bands", `infb_bookmc ${tag} (true bands v2)`, 3,
         `infbands/infb_bookmc_${tag}_capability.done`, file, "infb_bookmc");
    if (fs.existsSync(path.join(EVAL, file + ".soup-invalid")))
      W[W.length - 1].state = "invalid";
  }
  push("synth-384k", "synthetic >128k", "counting+cwe @384k (n=3)", 6,
       "infbands/synth_384k.done", "synth_384k.jsonl", "--ctx 384000");
  push("synth-512k", "synthetic >128k", "counting+cwe @512k (n=3)", 6,
       "infbands/synth_512k.done", "synth_512k.jsonl", "--ctx 512000");
  push("synth-512k-greedy", "synthetic >128k", "counting @512k greedy (n=3)", 3,
       "infbands/synth_512k_greedy_counting.done", "synth_512k_greedy_counting.jsonl",
       "--mode parity --ctx 512000");
  push("ruler", "RULER", "RULER 4 tasks @128k-512k (n=3)", 48,
       "ruler-gt128k.done", "ruler_gt128k.jsonl", "--plugin ruler");
  push("mrcr", "MRCR v2", "MRCR mrcr2/mrcr4 @131k+262k (n=3, official data)", 12,
       "mrcr-gt128k.done", "mrcr_gt128k.jsonl", "--plugin mrcr");
  push("enrich", "weak-axis enrich", "stock weak-axis n=5 (pre-goal, reused)", 20,
       "stage4-stockweak5.done", "stock_weak5.jsonl", null);
  push("greedy-confirm", "weak-axis enrich", "greedy confirm stock (reused)", 20,
       "greedy-confirm-stock.done", "confirm_greedy_stock.jsonl", null);
  push("greedy-confirm-qk43", "weak-axis enrich", "greedy confirm qk4.3 (reused)", 20,
       "greedy-confirm-qk4.3.done", "confirm_greedy_qk4.3.jsonl", null);
  const qk50n = countRows("confirm_greedy_qk5.0.jsonl");
  W.push({ id: "greedy-confirm-qk50", group: "weak-axis enrich",
           label: "greedy confirm qk5.0 — cancelled (redundant: verdict 3x-supported)",
           total: 20, rows: qk50n, state: qk50n >= 20 ? "complete" : "cancelled" });

  // queued chains: scripts in wait loops awaiting predecessors
  const chains = [
    ["ruler_gt128k.sh", "ruler (chained)", mk("synth-gt128k.done") ? "waiting GPU" : "armed"],
    ["infb_bands2.sh", "bookmc true bands v2", mk("ruler-gt128k.done") ? "waiting GPU" : "armed"],
    ["mrcr_gt128k.sh", "MRCR v2 lanes", mk("ruler-gt128k.done") ? "waiting GPU" : "armed"],
    ["synth_gt128k.sh", "synthetic >128k", "running"],
  ];
  for (const [script, label, state] of chains) {
    let alive = false;
    try { alive = execSync("pgrep -f " + script, { timeout: 4000, encoding: "utf8" }).trim().length > 0; } catch {}
    W.push({ id: "chain-" + script, group: "queues", label: script + " — " + label,
             total: null, rows: null, state: alive ? state : "not running" });
  }
  res.json({ ts: new Date().toISOString(), workloads: W });
});

app.get("/api/infb", (req, res) => {
  /* InfBench honest-length bands (goal afe6584b): infb_infb_<task>_<band>_<mode>.jsonl
   * grouped by (task, band, mode) with true-token ranges from expected.ctx_tokens. */
  const fs = require("fs");
  const out = [];
  for (const f of fs.readdirSync(EVAL).sort()) {
    const m = f.match(/^infb_infb_([a-z]+)_(\d+-\d+k)_(capability|parity)\.jsonl$/);
    if (!m) continue;
    try {
      const rows = fs.readFileSync(path.join(EVAL, f), "utf8").trim().split("\n")
        .filter(Boolean).map((l) => JSON.parse(l)).filter((r) => !r.error);
      if (!rows.length) continue;
      const toks = rows.map((r) => r.expected && r.expected.ctx_tokens).filter(Boolean);
      out.push({ task: "infb_" + m[1], band: m[2], mode: m[3], n: rows.length,
                 hits: rows.reduce((a, r) => a + r.score, 0),
                 mean: +(rows.reduce((a, r) => a + r.score, 0) / rows.length).toFixed(3),
                 trueMin: toks.length ? Math.min(...toks) : null,
                 trueMax: toks.length ? Math.max(...toks) : null });
    } catch { /* skip unreadable */ }
  }
  res.json(out);
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
