# rag-eval-starter

A minimal, runnable evaluation framework for a Chroma-backed RAG pipeline,
scored with **Ragas** (exploration) and **DeepEval** (CI gate). It implements
the four core RAG metrics and wires the DeepEval suite into GitHub Actions so
regressions are blocked before they merge.

## How the pieces map

```
data/eval_dataset.json   labelled cases (question + ground-truth answer)
rag/corpus.py            sample knowledge base (swap for your chunks)
rag/pipeline.py          Chroma ingest + retrieve + grounded generation  <- system under test
llm.py                   provider-flexible LLM / embeddings / judge wrappers
evals/run_pipeline.py    runs the dataset through the pipeline -> scorable records
evals/ragas_explore.py   notebook-style scoring + per-tag rollup       (tune here)
evals/metrics.py         custom RefusalCorrectnessMetric + shared metric-set selection
evals/test_rag_deepeval.py  pytest single-run gate                     (debug here)
evals/regression_gate.py    N-runs + pass-rate vs baseline             (block here)
evals/baseline.json      committed pass-rate baseline (the spec)
.github/workflows/eval.yml  runs the gate on every PR
config.py                providers, models, top_k, thresholds, REPEATS, TOLERANCE
```

The pipeline is the *system under test*; Ragas/DeepEval are the *scorers*; the
metrics are *what they compute*; the workflow file is the *gate*.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # add your OPENAI_API_KEY
set -a; source .env; set +a

python -m rag.pipeline                # smoke-test ingest + one answer
python -m evals.ragas_explore         # score everything + diagnostic read
pytest evals/test_rag_deepeval.py -v  # the gate
```

Defaults run everything on OpenAI with one key. To answer and/or judge with
Claude, set `LLM_PROVIDER=anthropic` / `JUDGE_PROVIDER=anthropic` and add
`ANTHROPIC_API_KEY` (embeddings stay on OpenAI or set `EMBED_PROVIDER=local`).

## The two-stage workflow

1. **Explore (Ragas).** Iterate on chunking, `TOP_K`, prompt, model. Read the
   metric *combination*, not single numbers:
   - high context precision + **low recall** -> retrieval is incomplete (missing
     chunks): raise `TOP_K`, improve chunking, add a reranker.
   - **low faithfulness** + ok recall -> the content was retrieved but the answer
     isn't grounded: a generation problem (tighten prompt, lower temperature).
   - everything high but answers still wrong -> the index is stale. The metrics
     verify the answer is faithful *to the chunk*, not that the chunk is current.
2. **Gate (DeepEval).** Once metrics stabilise, the same four metrics run as
   pytest assertions in CI and block merges below threshold.

## The four metrics

| Metric | Surface | Question it answers |
|---|---|---|
| Faithfulness | generation | Is the answer supported by the retrieved context? |
| Answer relevancy | generation | Does the answer address the question? |
| Context precision | retrieval | Are the retrieved chunks relevant? |
| Context recall | retrieval | Did retrieval find everything needed (vs. the reference)? |

All four are LLM-judged, so budget ~0.1-0.3c per case and **calibrate the judge
against human labels before trusting it**. The dataset also includes
out-of-corpus cases whose reference is "I don't know..." to catch hallucination.

## Findings from the first runs

Three findings from running the dataset end-to-end (gpt-4o-mini as generator,
`claude-sonnet-4-6` as judge). These shaped the roadmap below.

**1. Standard RAG metrics penalise correct refusals.** The dataset includes
two out-of-corpus cases whose reference answer is `"I don't know based on the
available information."` The pipeline correctly refused both — exactly the
behaviour we want — and was scored:

| Metric | Score on refusal cases | Why |
|---|---|---|
| Faithfulness | 1.0 ✓ | "I don't know" makes no factual claims to check |
| Answer relevancy (Ragas) | 0.0 ✗ | Reverse-engineers a question from the answer; a refusal doesn't map back to the original |
| Context precision (DeepEval) | 0.0 ✗ | No retrieved chunks are relevant to an unanswerable question |
| Context recall (DeepEval) | 0.0 ✗ | Tries to attribute the reference ("I don't know...") to retrieved chunks; not there |

None of the four standard metrics have a concept of "correctly chose not to
answer." This isn't a configuration bug; it's an architectural gap in the
RAG-evaluation toolkit. **Both fixes are implemented in this repo:**

- `RefusalCorrectnessMetric` in `evals/metrics.py` — judge-free, binary,
  symmetric (catches both hallucination and false refusal). `test_rag_deepeval.py`
  applies tag-appropriate metric sets via `_metrics_for()`: refusal cases get
  Faithfulness + RefusalCorrectness only; substantive cases get the four
  standard metrics + RefusalCorrectness as a false-refusal safety net.
- **Per-tag metric breakdown** in `evals/ragas_explore.py` — `means_by_tag()`
  rolls up scores by tag (`happy-path` / `should-refuse` / `multi-hop` / etc.)
  so refusal cases stop polluting aggregate numbers. The diagnostic read now
  segments by tag rather than reporting on the aggregate.

Result: the DeepEval gate went from **7/9 to 9/9 pass** without lowering any
threshold, and the Ragas exploration's diagnostic read now correctly flags
"refusal cases structurally low" instead of declaring the whole pipeline
healthy from a misleading aggregate.

**2. Ragas and DeepEval disagree on "the same" metric.** On the `multi-hop`
case, Ragas' `LLMContextPrecisionWithReference` scored `0.0` while DeepEval's
`ContextualPrecisionMetric` passed. Same case, same retrieved chunks, same
reference. Ragas penalises rank order (relevant chunks at position 1 score
higher than at position 3); DeepEval is more lenient. **Pick one tool per
workflow stage and stick to it** — don't cross-compare scores between libraries.

**3. DeepEval's reasoning output is the killer feature.** Ragas gives you
numbers; DeepEval gives you the judge's natural-language explanation for every
score. Example failure reason from the live run:

> *"The score is 0.00 because the expected output states 'I don't know based
> on the available information,' but sentence 1 cannot be attributed to any
> nodes in retrieval context, as the retrieval context actually contains
> detailed and relevant information that would directly answer the question,
> making the claim of uncertainty unsupported."*

That reasoning revealed the judge was itself slightly confused (the Pro plan
chunk does *not* answer "what discount do students get"). Without DeepEval's
reasoning surface you'd never catch judge-reasoning artifacts like this.

## The real regression gate

The naive gate (`test_rag_deepeval.py`) runs each case once and fails on any
single sub-threshold metric. That's brittle: LLM judges are noisy (we've
observed Faithfulness flip 1.0 → 0.5 between runs on identical input), so a
clean run can fail purely by chance, and a real regression can hide inside
one lucky run.

`evals/regression_gate.py` is the statistically-sound version:

1. Run each case `REPEATS` times (default 3, configurable via env).
2. For each (tag, metric), compute pass-rate = fraction of (case, run) pairs
   where the metric passed its threshold.
3. Compare current pass-rates to the committed baseline at
   `evals/baseline.json`.
4. Fail only when a pass-rate drops more than `REGRESSION_TOLERANCE` (default
   0.10) below baseline. That's the statistical definition of a regression.

Usage:

```bash
python -m evals.regression_gate                    # check against baseline
python -m evals.regression_gate --update-baseline  # rewrite baseline.json (intentional)
python -m evals.regression_gate --repeats 5        # override N
```

`baseline.json` is committed to git — it's part of the spec. Update
intentionally, like a snapshot test. The pytest gate
(`test_rag_deepeval.py`) is now a debugging tool for asking "which metric is
failing on case X?"; the regression gate is the merge blocker.

**Pinning everything that defines a run** is what makes a regression
*attributable*: `config.py` for model/threshold knobs, `requirements.txt` for
library versions, the system prompt inlined in `rag/pipeline.py`,
`data/eval_dataset.json` for the test cases, `baseline.json` for the
expected pass-rates. Change one knob → re-run → diff a single
baseline-comparison output.

## Judge calibration

Every metric in this repo is computed by an LLM judging another LLM. If the
judge is miscalibrated, every score is proportionally wrong. `evals/calibration.py`
exists to ground the framework against human assessment:

```bash
python -m evals.calibration export     # runs eval, writes calibration_labels.json
# ... open the JSON, fill in `human_scores` (0.0-1.0) for each case x metric ...
python -m evals.calibration analyze    # MAE, Pearson r, bias per metric; flag anomalies
```

Labelling all 9 cases × ~4 metrics = 39 judgements, ~20 min for a focused
reviewer. The goal isn't statistical significance at this scale — it's
revealing **systematic bias**, e.g. "judge always rates relevancy +0.10 above
human" or "judge misses rank-order quality on multi-hop." When found,
remediations in order of cost: tighten the metric prompt; apply a per-metric
offset in the regression gate; switch judge model.

`evals/calibration_labels.json` IS committed to git — it's part of the
project's evidence. An empty file means "calibration is pending"; a filled-in
file is the audit trail.

## Caveats

- Library APIs (especially Ragas and DeepEval custom models) move fast — see the
  inline notes in `llm.py`. `requirements.txt` is frozen to the exact versions
  that produced the findings above; bump deliberately.
- Thresholds (0.80 / 0.75 / 0.70 / 0.70) are placeholders. They pass for
  happy-path cases out of the box but, per finding (1), guarantee refusal cases
  fail the gate — that needs to be fixed in the metric layer, not by lowering
  thresholds.
- Pinned to Python 3.11 — `chroma-hnswlib` has no pre-built wheel for 3.13 on
  Windows yet, and 3.11 matches what CI runs.
- Judge calibration against human labels hasn't been done. Treat absolute scores
  as advisory; trust *relative changes* between runs.
