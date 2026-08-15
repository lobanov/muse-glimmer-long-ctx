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
  { name: "LongBench v2", file: "suite_longbench_v2.jsonl", total: 45 },
  { name: "LongCodeQA", file: "suite_longcodeqa.jsonl", total: 45 },
  { name: "InfBench", file: "suite_infbench.jsonl", total: 54 },
  { name: "synth3 fill-in", file: "suite_synth3.jsonl", total: 189 },
  { name: "agentmem", file: "suite_agentmem.jsonl", total: 216 },
  { name: "run1 (§8 trained)", file: "run1_vllm.jsonl", total: 216 },
];

app.get("/api/status", async (req, res) => {
  const grids = GRIDS.map((g) => {
    const p = path.join(EVAL, g.file);
    return { ...g, rows: readJSONL(p).length, exists: fs.existsSync(p), mtime: mtime(p) };
  });
  const watchers = {};
  for (const w of WATCHERS) {
    const r = await probe("pgrep", ["-f", `[s]cripts/${w}`]);
    watchers[w] = r.out.length > 0;
  }
  const markers = Object.fromEntries(MARKERS.map((m) => [
    m, fs.existsSync(path.join(LOGS, m)) ?
      fs.readFileSync(path.join(LOGS, m), "utf8").trim() : null]));
  const tails = {};
  for (const t of ["eval-stock-gt128k", "eval-stock-cwe", "suite-grid-suite_nolima",
    "overnight-queue", "suite-lane", "train-run1", "stage4-queue", "stage7-queue"]) {
    const lines = tailFile(path.join(LOGS, `${t}.log`), 4);
    if (lines) tails[t] = lines;
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
    grids, watchers, markers, tails, corpus,
    docker: { ps: dockerPs.out, vllm_up: vllm.ok && vllm.out.includes("muse-glimmer") },
  });
});

app.get("/api/grid/:file", (req, res) => {
  const allowed = GRIDS.some((g) => g.file === req.params.file);
  if (!allowed) return res.status(400).json({ error: "unknown grid" });
  res.json(grid([req.params.file]));
});

app.listen(PORT, "0.0.0.0", () =>
  console.log(`dashboard: http://<host>:${PORT}  (root: ${ROOT})`));
