"""Ragas exploration (the notebook-style, non-gating half of the workflow).

Run this while tuning: it scores the whole dataset on the four core metrics,
prints a per-case table, a per-tag rollup, and an automatic diagnostic read.
Use it to decide WHAT to fix; use the DeepEval suite to GATE once the metrics
have stabilised.

    python -m evals.ragas_explore

Per-tag is the *primary* diagnostic view. Aggregate means mix happy-path and
should-refuse cases together, which hides the refusal-penalty quirk (see
README "Findings from the first runs"). Always read per-tag first.
"""
import pandas as pd
from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from evals.run_pipeline import build_records
from llm import get_ragas_judge


def main():
    records = build_records()
    eval_dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": r["user_input"],
                "retrieved_contexts": r["retrieved_contexts"],
                "response": r["response"],
                "reference": r["reference"],
            }
            for r in records
        ]
    )

    judge_llm, judge_embeddings = get_ragas_judge()
    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    df = result.to_pandas()
    df.insert(0, "id", [r["id"] for r in records])
    df.insert(1, "tags", [", ".join(r.get("tags", [])) for r in records])

    pd.set_option("display.max_colwidth", 40)
    print("\n=== Per-case scores ===")
    print(df.to_string(index=False))

    print("\n=== Aggregate means (read per-tag below first -- aggregates can mislead) ===")
    print(df.mean(numeric_only=True).to_string())

    per_tag = means_by_tag(records, df)
    print("\n=== Means by tag (the diagnostic view) ===")
    print(per_tag.to_string())

    diagnose(per_tag)


def means_by_tag(records, df):
    """One row per tag. Cases with multiple tags contribute to each group."""
    exploded = pd.DataFrame(
        [
            {"id": r["id"], "tag": tag}
            for r in records
            for tag in r.get("tags", [])
        ]
    )
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    merged = exploded.merge(df, on="id")
    out = merged.groupby("tag")[numeric_cols].mean()
    out.insert(0, "n", merged.groupby("tag").size())
    return out.sort_values("n", ascending=False)


def diagnose(per_tag):
    """Per-tag diagnostic. Aggregate means mix fundamentally different case
    types (happy-path vs. should-refuse); the per-tag view doesn't.
    """
    print("\n=== Diagnostic read ===")

    if "should-refuse" in per_tag.index:
        rr = per_tag.loc["should-refuse"]
        print(
            f"- Refusal cases: relevancy={rr.get('answer_relevancy', float('nan')):.2f}, "
            f"precision={rr.get('llm_context_precision_with_reference', float('nan')):.2f}.\n"
            f"  These are STRUCTURALLY LOW -- Ragas metrics can't score correct refusals.\n"
            f"  The DeepEval gate uses RefusalCorrectnessMetric (evals/metrics.py) instead."
        )

    if "happy-path" in per_tag.index:
        hp = per_tag.loc["happy-path"]
        precision = hp.get("llm_context_precision_with_reference", float("nan"))
        recall = hp.get("context_recall", float("nan"))
        faith = hp.get("faithfulness", float("nan"))

        if precision >= 0.7 and recall < 0.7:
            print("- Happy-path retrieval INCOMPLETE: high precision, low recall.")
            print("  Try larger top_k, better chunking, or a reranker.")
        elif faith < 0.7 and recall >= 0.7:
            print("- Happy-path generation NOT GROUNDED: low faithfulness, ok recall.")
            print("  Tighten the prompt / lower temperature.")
        elif precision >= 0.7 and recall >= 0.7 and faith >= 0.7:
            print("- Happy-path healthy. Remaining risk is the index itself")
            print("  (staleness/freshness -- metrics can't see this).")


if __name__ == "__main__":
    main()
