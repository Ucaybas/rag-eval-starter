# Handoff — rag-eval-starter

Context primer for picking this up in a code editor (e.g. Claude Code). Paste
this in, or just open the repo — paths below are repo-relative.

## Goal / background

Building an **evaluation framework for a RAG product**. The target role is
"design and implement evaluation frameworks for agentic and generative AI —
regression suites for prompts, models, retrieval quality, tool use, and
end-to-end agent behaviour." In practice the working stack is RAG-centric:
**Python, ChromaDB, RAG pipelines, DeepEval, Ragas, LLM-judged metrics**. This
repo is a runnable starter for exactly that.

## Mental model

A regression suite for AI is **statistical, not binary** — outputs vary run to
run, so a regression is a *meaningful drop in pass-rate vs. a baseline*, not any
single failing case. The framework is one harness that holds the pipeline fixed
except the one variable under test (model / prompt / top_k / embeddings).

Pipeline = **system under test**. Ragas + DeepEval = **scorers**. The four
metrics = **what they compute**. The workflow file = **the gate**.

## What's in the repo

```
config.py                providers, models, top_k, thresholds (versioned knobs)
llm.py                   provider-flexible LLM / embeddings / judge wrappers
rag/corpus.py            8-doc sample knowledge base (fictional product "Orbit")
rag/pipeline.py          Chroma ingest + retrieve + grounded generation (RagPipeline)
data/eval_dataset.json   9 labelled cases incl. 2 out-of-corpus "should refuse" cases
evals/run_pipeline.py    runs dataset through pipeline -> scorable records (supports repeats=N)
evals/ragas_explore.py   notebook-style scoring + diagnostic read (tune here)
evals/test_rag_deepeval.py  pytest gate, 4 metrics w/ thresholds (block here)
.github/workflows/eval.yml  runs the gate on PRs (needs OPENAI/ANTHROPIC secrets)
```

## Key decisions / conventions

- **Two-stage workflow:** Ragas for exploration (notebook ergonomics), DeepEval
  for the CI gate (pytest). Same four metrics in both.
- **Four core metrics:** faithfulness + answer relevancy (generation surface);
  context precision + context recall (retrieval surface).
- **Diagnostic reading** is the point, not single numbers:
  high precision + low recall = incomplete retrieval;
  low faithfulness + ok recall = ungrounded generation;
  all high but wrong = stale index (metrics can't see freshness).
- **Provider-flexible, OpenAI by default** (one key runs everything).
  `LLM_PROVIDER` / `JUDGE_PROVIDER` = openai|anthropic; `EMBED_PROVIDER` =
  openai|local. Embeddings stay OpenAI/local since Anthropic has no embeddings API.
- Everything config-driven via `config.py` + env (see `.env.example`).
- Default models: gpt-4o-mini (gen + judge), text-embedding-3-small. Anthropic
  path defaults: claude-haiku-4-5-20251001 (gen), claude-sonnet-4-6 (judge).
- Default thresholds: faithfulness 0.80, answer_relevancy 0.75, context
  precision 0.70, context recall 0.70 — **placeholders, not validated**.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env            # add OPENAI_API_KEY
set -a; source .env; set +a
python -m rag.pipeline                # smoke test
python -m evals.ragas_explore         # score + diagnostic read
pytest evals/test_rag_deepeval.py -v  # the gate
```

## Status (verified runnable as of 2026-05-23)

End-to-end runnable. Two latent bugs in the starter were fixed during first
install; `requirements.txt` is now frozen to the exact versions that produced
the results below.

**Fixed during first install:**

- `config.py` collection-name default was `"kb"` (2 chars), rejected by
  ChromaDB ≥0.5 (requires 3–63). Changed default to `"orbit_kb"`.
- `chromadb 0.5.x`'s bundled `OpenAIEmbeddingFunction` calls the removed
  pre-1.0 `openai.Embedding` API and is broken against `openai>=1.0`. Replaced
  with a small `_OpenAIEmbeddingFunction` wrapper in `llm.py` that uses the
  modern `OpenAI()` client.
- Python pinned to **3.11**: `chroma-hnswlib` has no Windows wheel for 3.13,
  and 3.11 matches the CI workflow.
- `AnthropicJudge.generate()` in `llm.py` *did not* hit the `schema=` TypeError
  on `deepeval==4.0.2` — the inline warning can stay as historical context but
  isn't currently a live concern.

**First-run results (single run, 9 cases):**

| Metric | Mean | Notes |
|---|---|---|
| Faithfulness | 1.000 | No hallucination — the "answer using ONLY context" prompt works |
| Context recall | 1.000 (Ragas) / 0.78 (DeepEval) | Disagreement driven by refusal cases (see below) |
| Answer relevancy | 0.728 | Dragged down by refusal cases scoring 0.0 |
| Context precision | 0.759 (Ragas) | Multi-hop scored 0.0 in Ragas (rank-order penalty), passed in DeepEval |

DeepEval gate: 7 of 9 cases pass. Both failures are the `should-refuse` cases,
failing on Contextual Precision + Recall — see "Refusal-penalty finding" below.

**Refusal-penalty finding (the headline result):**

All four standard RAG metrics, across both Ragas and DeepEval, score *correct
refusals* (e.g. "I don't know...") as 0.0. None of them have a concept of "the
system correctly chose not to answer." This is an architectural gap in the
standard toolkit, not a config bug. Full writeup is in `README.md` ("Findings
from the first runs").

**Fixes shipped:**

- `evals/metrics.py` defines `RefusalCorrectnessMetric` — judge-free, binary,
  symmetric (catches both hallucination and false refusal) — plus
  `metrics_for_record(record)`, the tag-appropriate metric-set selector
  shared by both the pytest gate and the regression gate.
- `evals/ragas_explore.py` now prints a **per-tag means rollup** alongside the
  per-case table; the diagnostic read is tag-aware (refusal cases flagged as
  structurally-low, "healthy" verdict gated on `happy-path` only).
- `evals/regression_gate.py` is the **real merge gate**: runs each case N
  times (default 3), computes per-tag pass-rates, compares to the committed
  baseline at `evals/baseline.json`, fails only when pass-rate drops
  more than `REGRESSION_TOLERANCE` (default 0.10) below baseline. `compare()`
  was unit-checked against four synthetic scenarios (no-regression-noise /
  real-regression / hallucination-starts / missing-tag) and behaves correctly
  on all four. End-to-end run against the baseline: clean `[PASS]`.
  `.github/workflows/eval.yml` now runs `python -m evals.regression_gate`
  instead of pytest. The pytest gate (`test_rag_deepeval.py`) is preserved
  as a single-case debug tool.
- `evals/calibration.py` provides judge calibration infrastructure:
  `export` runs the eval once and writes `evals/calibration_labels.json`
  with judge scores + empty human-score slots; `analyze` computes
  per-metric MAE, Pearson correlation, and bias (judge mean - human mean)
  and flags per-case anomalies (delta > 0.20). The labels file is committed
  to git as audit trail (empty = pending; filled = grounded). Tool is built
  and exported; **human labelling is the remaining manual step** -- by design,
  not something a second LLM can substitute for.

**Other notes:**

- `EMBED_PROVIDER=local` pulls sentence-transformers/torch and downloads a model.
- ChromaDB telemetry prints harmless warnings (`Failed to send telemetry event...`)
  due to a posthog SDK version mismatch — silence with
  `Settings(anonymized_telemetry=False)` on the client if it's noisy.
- A segfault (exit 139) occurs after Ragas exploration completes — happens
  during chromadb/onnxruntime teardown, all data is captured first. Harmless
  interactively; would matter if piping output.

## Next steps (priority order)

Reordered after first-run findings — the refusal-penalty result elevates
metric-layer work above the regression-gate scaffolding.

1. **Fill in `evals/calibration_labels.json`** (manual, ~20 min): score each
   case×metric in `human_scores` (0.0-1.0). Then run
   `python -m evals.calibration analyze` to see MAE/correlation/bias per metric.
   This is the only remaining manual step to ground the whole framework.
2. **Unit tests for `evals/metrics.py` and `evals/regression_gate.py`:** the
   spot-check confirmed `RefusalCorrectness` catches both failure modes and
   `compare()` flags regressions correctly. Promote both into proper
   `evals/test_metrics.py` and `evals/test_regression_gate.py` so the
   eval infrastructure is itself regression-tested.
3. **Pure-IR retrieval metrics:** add Precision@K / Recall@K / MRR (needs labelled
   relevant-doc IDs per query) for sharper, judge-free retrieval diagnosis.
4. **Bump baseline confidence:** the current baseline was generated at N=3 (~$0.50
   per generation). Once the dataset/pipeline is stable, regenerate at N=5 or
   N=10 (`--repeats 5`) for a tighter, more statistically-grounded baseline.
5. **Wire to a real corpus** (deferred — this repo is a portfolio piece; the
   fictional Orbit corpus is fine because it gives unambiguous ground truth).
6. **Production loop** (deferred — no production traffic to mine from).
