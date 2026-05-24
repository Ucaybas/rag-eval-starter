"""Regression gate -- the statistical version of the eval.

The naive gate (test_rag_deepeval.py) runs each case once and fails on any
metric below threshold. That's brittle: LLM judges are noisy (we've observed
the same input flip faithfulness 1.0 -> 0.5 between runs), so a clean run can
fail purely by chance, and a real regression can hide inside one lucky run.

This gate fixes both:
  1. Run each case N times (config.REPEATS).
  2. For each (tag, metric), compute pass-rate = fraction of (case, run) pairs
     where the metric passed its threshold.
  3. Compare current pass-rates to a committed baseline JSON.
  4. Fail only when a pass-rate drops more than REGRESSION_TOLERANCE below
     baseline. That's the statistical definition of a regression.

Usage:
    python -m evals.regression_gate                    # check against baseline
    python -m evals.regression_gate --update-baseline  # rewrite baseline.json
    python -m evals.regression_gate --repeats 5        # override N

Baseline lives at evals/baseline.json and IS committed to git -- it's part of
the spec. Update intentionally, like a snapshot test.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from deepeval.test_case import LLMTestCase

import config
from evals.metrics import metrics_for_record
from evals.run_pipeline import build_records

BASELINE_PATH = Path(__file__).parent / "baseline.json"


def run_eval(repeats: int) -> pd.DataFrame:
    """Score every (record x metric) pair. Long-format DataFrame:
    columns = [id, tag, metric, run, score, passed].

    Each case is exploded across its tags so a case with [happy-path, pricing]
    contributes to both groups in the per-tag rollup.
    """
    records = build_records(repeats=repeats)
    rows = []
    for r in records:
        test_case = LLMTestCase(
            input=r["user_input"],
            actual_output=r["response"],
            expected_output=r["reference"],
            retrieval_context=r["retrieved_contexts"],
        )
        for m in metrics_for_record(r):
            # async_mode=False so we can call measure() synchronously without
            # an event loop. The async path is for parallel scoring inside
            # evaluate()/assert_test(); we drive the loop ourselves here.
            m.async_mode = False
            score = m.measure(test_case)
            passed = bool(m.is_successful())
            for tag in r.get("tags", []) or ["untagged"]:
                rows.append(
                    {
                        "id": r["id"],
                        "tag": tag,
                        "metric": m.__name__,
                        "run": r["run"],
                        "score": score,
                        "passed": passed,
                    }
                )
    return pd.DataFrame(rows)


def pass_rate_by_tag(df: pd.DataFrame) -> dict:
    """{tag: {metric: pass_rate}} -- the shape that gets compared to baseline."""
    grouped = df.groupby(["tag", "metric"])["passed"].mean().reset_index()
    out: dict = {}
    for _, row in grouped.iterrows():
        out.setdefault(row["tag"], {})[row["metric"]] = round(float(row["passed"]), 4)
    return out


def compare(current: dict, baseline: dict, tolerance: float) -> list[dict]:
    """Return per-(tag, metric) regressions where current dropped > tolerance
    below baseline. Empty list = clean run.

    New tag/metric combinations (in current but not baseline) are ignored --
    they aren't regressions, just additions.
    """
    regressions = []
    for tag, metric_map in baseline.items():
        for metric, base_rate in metric_map.items():
            cur_rate = current.get(tag, {}).get(metric)
            if cur_rate is None:
                # Metric/tag was in baseline but isn't in current run --
                # likely a dataset/metric-set change. Flag separately.
                regressions.append(
                    {
                        "tag": tag,
                        "metric": metric,
                        "baseline": base_rate,
                        "current": None,
                        "delta": None,
                        "kind": "MISSING",
                    }
                )
                continue
            if cur_rate < base_rate - tolerance:
                regressions.append(
                    {
                        "tag": tag,
                        "metric": metric,
                        "baseline": base_rate,
                        "current": cur_rate,
                        "delta": cur_rate - base_rate,
                        "kind": "REGRESSION",
                    }
                )
    return regressions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Rewrite baseline.json with current run's pass rates (no comparison).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=config.REPEATS,
        help=f"Runs per case (default {config.REPEATS}).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=config.REGRESSION_TOLERANCE,
        help=f"Allowed pass-rate drop (default {config.REGRESSION_TOLERANCE}).",
    )
    args = parser.parse_args()

    print(f"Running eval: {args.repeats} repeats per case ...")
    df = run_eval(args.repeats)
    current = pass_rate_by_tag(df)

    print("\n=== Current pass-rates by tag x metric ===")
    print(json.dumps(current, indent=2, sort_keys=True))

    if args.update_baseline:
        BASELINE_PATH.write_text(json.dumps(current, indent=2, sort_keys=True))
        print(f"\nBaseline written to {BASELINE_PATH.relative_to(Path.cwd())}")
        return 0

    if not BASELINE_PATH.exists():
        print(
            f"\nNo baseline at {BASELINE_PATH}. "
            "Run with --update-baseline first to seed it."
        )
        return 2

    baseline = json.loads(BASELINE_PATH.read_text())
    regressions = compare(current, baseline, args.tolerance)

    if not regressions:
        print(f"\n[PASS] No regressions (tolerance={args.tolerance}).")
        return 0

    print(f"\n[FAIL] {len(regressions)} regression(s) (tolerance={args.tolerance}):")
    print(f"  {'tag':16s} {'metric':28s} {'baseline':>9s} {'current':>9s} {'delta':>8s}")
    for r in regressions:
        cur_str = f"{r['current']:.2f}" if r["current"] is not None else "MISSING"
        delta_str = f"{r['delta']:+.2f}" if r["delta"] is not None else "    --"
        print(
            f"  {r['tag']:16s} {r['metric']:28s} "
            f"{r['baseline']:>9.2f} {cur_str:>9s} {delta_str:>8s}"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
