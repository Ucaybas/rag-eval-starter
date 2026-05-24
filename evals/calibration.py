"""Judge calibration -- ground the framework against human assessment.

Everything this repo reports comes from an LLM judging another LLM. If the
judge is miscalibrated, every score is proportionally wrong and our regression
gate is comparing noise to noise. Calibration is the step that gives the
numbers meaning.

Workflow:
  1. python -m evals.calibration export
       Runs the eval once, writes evals/calibration_labels.json with the
       judge's score for each (case, metric) pair and an empty `human_score`
       slot to fill in.
  2. Open evals/calibration_labels.json. For each case, read the question,
       retrieved_contexts, response, and reference, then score each metric
       in `human_scores` (0.0 to 1.0). Add notes if useful.
  3. python -m evals.calibration analyze
       Computes per-metric agreement stats (MAE, Pearson correlation, mean
       judge vs. mean human) over the cases you've labelled, and flags
       individual cases where judge<->human disagreement exceeds 0.20.

Labelling all 9 cases x ~4 metrics = 36 judgements -- about 20 minutes for a
focused human reviewer. The goal isn't statistical significance at this scale,
it's revealing systematic bias: "the judge always rates relevancy 0.10 above
me" or "the judge misses rank-order quality on multi-hop retrieval."

If you find consistent bias, three remediations (in order of cost):
  - Tighten the metric prompt (some Ragas/DeepEval metrics expose a prompt
    template you can override).
  - Apply a per-metric offset in the regression gate.
  - Switch judge model (e.g. claude-opus-4-5 instead of claude-sonnet-4-6).
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

from deepeval.test_case import LLMTestCase

from evals.metrics import metrics_for_record
from evals.run_pipeline import build_records

LABELS_PATH = Path(__file__).parent / "calibration_labels.json"
ANOMALY_THRESHOLD = 0.20


def export() -> int:
    """Run the eval once, write a labelling file with empty human_scores."""
    print(f"Running eval at repeats=1 for calibration export ...")
    records = list(build_records(repeats=1))

    cases = []
    for r in records:
        tc = LLMTestCase(
            input=r["user_input"],
            actual_output=r["response"],
            expected_output=r["reference"],
            retrieval_context=r["retrieved_contexts"],
        )
        judge_scores = {}
        for m in metrics_for_record(r):
            m.async_mode = False
            judge_scores[m.__name__] = round(m.measure(tc), 4)

        cases.append(
            {
                "id": r["id"],
                "question": r["user_input"],
                "reference": r["reference"],
                "response": r["response"],
                "retrieved_contexts": r["retrieved_contexts"],
                "tags": r.get("tags", []),
                "judge_scores": judge_scores,
                # Fill in 0.0-1.0 for each metric. Leave null to skip.
                "human_scores": {k: None for k in judge_scores},
                "notes": "",
            }
        )

    LABELS_PATH.write_text(
        json.dumps(
            {
                "description": (
                    "Judge calibration labels. For each case, read the "
                    "question/contexts/response/reference and score each metric "
                    "yourself in human_scores (0.0-1.0). Leave null to skip. "
                    "Then run: python -m evals.calibration analyze"
                ),
                "metric_definitions": {
                    "Faithfulness": "Is every claim in the response supported by retrieved_contexts? (1.0 = fully supported)",
                    "Answer Relevancy": "Does the response actually address the question? (1.0 = directly on point)",
                    "Contextual Precision": "Are the retrieved chunks relevant to the question, with most-relevant first? (1.0 = all relevant, well-ranked)",
                    "Contextual Recall": "Does the retrieved context contain everything needed for the reference answer? (1.0 = complete)",
                    "Refusal Correctness": "Did the refuse-or-answer decision match the reference? (1.0 = matched, 0.0 = mismatched)",
                },
                "cases": cases,
            },
            indent=2,
        )
    )

    n_judge = sum(len(c["judge_scores"]) for c in cases)
    print(f"\nWrote {LABELS_PATH.relative_to(Path.cwd())}")
    print(f"  {len(cases)} cases, {n_judge} judge scores, {n_judge} human slots to fill.")
    print(f"  Open the file, fill in human_scores, then run: python -m evals.calibration analyze")
    return 0


def analyze() -> int:
    """Compute agreement stats over the cases the human has labelled."""
    if not LABELS_PATH.exists():
        print(f"No labels file at {LABELS_PATH}.")
        print("Run `python -m evals.calibration export` first.")
        return 2

    data = json.loads(LABELS_PATH.read_text())
    cases = data["cases"]

    # Collect (judge, human, case_id) tuples grouped by metric, skipping unlabelled.
    by_metric: dict[str, list[tuple[float, float, str]]] = {}
    total_slots = 0
    labelled = 0
    for c in cases:
        for metric, judge_score in c["judge_scores"].items():
            total_slots += 1
            human_score = c.get("human_scores", {}).get(metric)
            if human_score is None:
                continue
            labelled += 1
            by_metric.setdefault(metric, []).append(
                (float(judge_score), float(human_score), c["id"])
            )

    pct = (100 * labelled / total_slots) if total_slots else 0
    print(f"\n=== Calibration coverage: {labelled}/{total_slots} ({pct:.0f}%) ===")

    if labelled == 0:
        print("\nNo human labels yet.")
        print(f"Open {LABELS_PATH.relative_to(Path.cwd())} and fill in `human_scores` (0.0-1.0).")
        return 1

    print(
        f"\n{'metric':25s} {'n':>3s} {'MAE':>7s} {'r':>7s} "
        f"{'judge_mean':>11s} {'human_mean':>11s} {'bias':>7s}"
    )
    for metric in sorted(by_metric):
        pairs = by_metric[metric]
        n = len(pairs)
        judges = [j for j, _, _ in pairs]
        humans = [h for _, h, _ in pairs]
        mae = statistics.mean(abs(j - h) for j, h, _ in pairs)
        bias = statistics.mean(j - h for j, h, _ in pairs)  # +ve = judge over-rates
        try:
            r = statistics.correlation(judges, humans) if n >= 2 and len(set(judges)) > 1 and len(set(humans)) > 1 else float("nan")
        except statistics.StatisticsError:
            r = float("nan")
        print(
            f"{metric:25s} {n:>3d} {mae:>7.3f} {r:>7.3f} "
            f"{statistics.mean(judges):>11.3f} {statistics.mean(humans):>11.3f} "
            f"{bias:>+7.3f}"
        )

    print(f"\n=== Cases with judge<->human gap > {ANOMALY_THRESHOLD} ===")
    flagged = []
    for metric, pairs in by_metric.items():
        for j, h, case_id in pairs:
            if abs(j - h) > ANOMALY_THRESHOLD:
                flagged.append((case_id, metric, j, h))
    if not flagged:
        print("None.")
    else:
        for case_id, metric, j, h in sorted(flagged):
            print(f"  {case_id:25s} {metric:25s} judge={j:.2f}  human={h:.2f}  delta={j-h:+.2f}")

    # Hint at next steps based on what we see.
    print("\n=== Interpretation ===")
    if labelled < total_slots:
        print(f"- Coverage is {pct:.0f}%. Label the rest for tighter estimates.")
    biggest_bias_metric = max(by_metric, key=lambda m: abs(statistics.mean(j - h for j, h, _ in by_metric[m])))
    biggest_bias = statistics.mean(j - h for j, h, _ in by_metric[biggest_bias_metric])
    if abs(biggest_bias) > 0.10:
        direction = "over-rates" if biggest_bias > 0 else "under-rates"
        print(
            f"- Largest systematic bias: judge {direction} `{biggest_bias_metric}` "
            f"by {abs(biggest_bias):.2f} on average."
        )
        print("  Remediations (cheapest first): tighten metric prompt, apply per-metric offset, switch judge model.")
    else:
        print("- No metric has systematic bias > 0.10. Judge looks reasonably calibrated.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export", help="Run eval and write calibration_labels.json")
    sub.add_parser("analyze", help="Compute judge-vs-human agreement stats")
    args = p.parse_args()

    if args.cmd == "export":
        return export()
    if args.cmd == "analyze":
        return analyze()
    return 0


if __name__ == "__main__":
    sys.exit(main())
