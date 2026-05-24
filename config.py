"""Central configuration, driven by environment variables.

Everything that defines a run lives here so it can be versioned and pinned:
provider, models, retrieval settings, and the regression thresholds the CI
gate enforces. Change a value -> re-run the suite -> compare to baseline.
"""
import os

# --- Providers -------------------------------------------------------------
# Generation = the LLM that answers questions inside the RAG pipeline.
# Judge      = the LLM that scores outputs (used by Ragas and DeepEval).
# Embeddings = used by Chroma for retrieval AND by Ragas answer-relevancy.
#
# Default to OpenAI everywhere so one API key runs the whole thing end to end.
# To judge/answer with Claude instead, set *_PROVIDER=anthropic (embeddings
# stay on OpenAI or local, since Anthropic has no first-party embeddings API).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")        # openai | anthropic
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "openai")    # openai | anthropic
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "openai")    # openai | local

GEN_MODEL = os.getenv(
    "GEN_MODEL",
    "claude-haiku-4-5-20251001" if LLM_PROVIDER == "anthropic" else "gpt-4o-mini",
)
JUDGE_MODEL = os.getenv(
    "JUDGE_MODEL",
    "claude-sonnet-4-6" if JUDGE_PROVIDER == "anthropic" else "gpt-4o-mini",
)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")

# --- Retrieval (the knobs that move retrieval metrics) ---------------------
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store")
COLLECTION = os.getenv("COLLECTION", "orbit_kb")
TOP_K = int(os.getenv("TOP_K", "3"))

# --- Data ------------------------------------------------------------------
DATASET_PATH = os.getenv("DATASET_PATH", "data/eval_dataset.json")

# --- Regression thresholds (the gate) --------------------------------------
# A case fails when a metric drops below its threshold. Starting points only;
# calibrate against your own labelled data. See README for the rationale.
THRESHOLDS = {
    "faithfulness": float(os.getenv("THR_FAITHFULNESS", "0.80")),
    "answer_relevancy": float(os.getenv("THR_ANSWER_RELEVANCY", "0.75")),
    "contextual_precision": float(os.getenv("THR_CONTEXT_PRECISION", "0.70")),
    "contextual_recall": float(os.getenv("THR_CONTEXT_RECALL", "0.70")),
}

# --- Regression gate (the statistical version of the gate) -----------------
# Run each case N times; compute pass-rate per tag*metric; fail when current
# pass-rate drops more than TOLERANCE below the committed baseline. This is
# the gate that should block merges (test_rag_deepeval.py is for debugging).
REPEATS = int(os.getenv("REPEATS", "3"))
REGRESSION_TOLERANCE = float(os.getenv("REGRESSION_TOLERANCE", "0.10"))
