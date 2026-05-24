"""Run the labelled dataset through the pipeline once, producing the records
that both Ragas and DeepEval consume. Optionally repeat each case N times so
downstream code can reason about variance (the statistical-regression point).
"""
import json

import config
from rag.pipeline import RagPipeline


def load_cases(path: str = config.DATASET_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)["cases"]


def build_records(repeats: int = 1, ensure_ingested: bool = True) -> list[dict]:
    """Return a list of records:
        {id, user_input, retrieved_contexts, response, reference, tags, run}
    matching Ragas' column names so it can be consumed directly.
    """
    pipe = RagPipeline()
    if ensure_ingested:
        pipe.ingest()

    records = []
    for case in load_cases():
        for run in range(repeats):
            answer, contexts = pipe.answer(case["question"])
            records.append(
                {
                    "id": case["id"],
                    "user_input": case["question"],
                    "retrieved_contexts": contexts,
                    "response": answer,
                    "reference": case["reference"],
                    "tags": case.get("tags", []),
                    "run": run,
                }
            )
    return records


if __name__ == "__main__":
    for r in build_records():
        print(f"[{r['id']}] {r['response'][:80]}")
