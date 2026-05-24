"""DeepEval suite -- single-run gate for debugging individual cases.

Each labelled case becomes an LLMTestCase scored on a tag-appropriate metric
set. A metric below its threshold fails the test.

    pytest evals/test_rag_deepeval.py -v

NOTE on purpose: this is now a *debugging* tool, not the merge gate. The real
gate is `evals/regression_gate.py`, which runs N times and compares pass-rates
to a committed baseline (the statistically-sound way to detect regressions in
a non-deterministic system). This file is useful when you want to ask "which
metric is failing on case X?" -- pytest's parametrize ergonomics are better
for that than the regression gate's aggregate output.

Metric-set selection lives in `evals/metrics.py:metrics_for_record()` so this
gate and the regression gate score cases identically.
"""
import functools

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from evals.metrics import metrics_for_record
from evals.run_pipeline import build_records


@functools.lru_cache(maxsize=1)
def _records():
    # Built once per session; lru_cache avoids re-querying the pipeline per case.
    return tuple(build_records())


@pytest.mark.parametrize("record", _records(), ids=lambda r: r["id"])
def test_rag_case(record):
    test_case = LLMTestCase(
        input=record["user_input"],
        actual_output=record["response"],
        expected_output=record["reference"],
        retrieval_context=record["retrieved_contexts"],
    )
    # assert_test runs every metric and fails if any is below threshold.
    assert_test(test_case, metrics_for_record(record))
