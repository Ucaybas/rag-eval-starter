"""Custom DeepEval metrics for RAG.

RefusalCorrectness fills a gap in the standard four RAG metrics: none of them
have a concept of "the system correctly chose not to answer." On out-of-corpus
questions, a correct refusal scores 0.0 on Contextual Precision and Recall
because there's nothing relevant to retrieve and the refusal phrase isn't in
any chunk. See README "Findings from the first runs" for the motivating data.

This metric is judge-free (string-pattern based) — exact, cheap, and not
subject to LLM-judge calibration drift. The trade-off is brittleness if the
refusal phrasing changes; broaden REFUSAL_PHRASES if needed.
"""
from deepeval.metrics import BaseMetric
from deepeval.metrics.base_metric import SingleTurnParams
from deepeval.test_case import LLMTestCase

# Substring matches (case-insensitive). The pipeline's system prompt asks
# the model to reply "I don't know based on the available information." — so
# the first two cover the intended path; the rest catch likely paraphrases.
REFUSAL_PHRASES = (
    "i don't know",
    "i do not know",
    "cannot answer",
    "no information",
    "not in the",
    "unable to find",
    "no relevant",
)


def is_refusal(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)


def metrics_for_record(record: dict) -> list:
    """Tag-appropriate metric set for a record.

    Refusal cases get Faithfulness + RefusalCorrectness only -- Contextual
    Precision/Recall always score 0 on correct refusals (no chunks to be
    relevant to an unanswerable question; refusal phrase isn't in any chunk).

    Substantive cases get the four standard RAG metrics + RefusalCorrectness
    as a false-refusal safety net.

    Shared between the pytest gate (test_rag_deepeval.py) and the regression
    gate (regression_gate.py) so they score cases identically.
    """
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )

    import config
    from llm import get_deepeval_model

    model = get_deepeval_model()
    t = config.THRESHOLDS
    faithfulness = FaithfulnessMetric(threshold=t["faithfulness"], model=model)
    refusal = RefusalCorrectnessMetric()

    if is_refusal(record["reference"]):
        return [faithfulness, refusal]

    return [
        faithfulness,
        AnswerRelevancyMetric(threshold=t["answer_relevancy"], model=model),
        ContextualPrecisionMetric(threshold=t["contextual_precision"], model=model),
        ContextualRecallMetric(threshold=t["contextual_recall"], model=model),
        refusal,
    ]


class RefusalCorrectnessMetric(BaseMetric):
    """Binary: did the system's refuse-or-answer decision match the reference?

    Symmetric — catches BOTH failure modes:
      - HALLUCINATION: reference is a refusal, system answered substantively (worst)
      - FALSE REFUSAL: reference is substantive, system refused (annoying)

    Otherwise scores 1.0. Threshold defaults to 1.0 (binary): the metric
    either fully passes or fully fails. No partial credit.
    """

    _required_params = [
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ]

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score = None
        self.reason = None
        self.success = None
        self.error = None
        self.strict_mode = False
        self.async_mode = False
        self.evaluation_cost = 0.0

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        ref_refuses = is_refusal(test_case.expected_output)
        out_refuses = is_refusal(test_case.actual_output)

        if ref_refuses and out_refuses:
            self.score = 1.0
            self.reason = "Correctly refused (reference is a refusal)."
        elif not ref_refuses and not out_refuses:
            self.score = 1.0
            self.reason = "Correctly answered substantively."
        elif ref_refuses and not out_refuses:
            self.score = 0.0
            self.reason = (
                "HALLUCINATION: reference is a refusal but system answered "
                f"substantively: {test_case.actual_output!r}"
            )
        else:
            self.score = 0.0
            self.reason = (
                "FALSE REFUSAL: reference is substantive but system refused. "
                f"Expected: {test_case.expected_output!r}"
            )

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self):
        return "Refusal Correctness"
