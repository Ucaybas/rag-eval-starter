"""Thin provider abstraction so the pipeline and judges are swappable.

Three concerns, kept separate on purpose:
  - get_embedding_function(): Chroma embedding function (retrieval).
  - generate():               the pipeline's answer-generation LLM.
  - get_ragas_judge() / get_deepeval_model(): the scoring LLMs.

Keeping the judge separate from the generator matters: if the same model both
answers and grades, you measure agreement, not quality.
"""
import config


# --- Embeddings (used by Chroma) -------------------------------------------
class _OpenAIEmbeddingFunction:
    """Modern-OpenAI-SDK embedding function for Chroma.

    Chroma 0.5.x ships an OpenAIEmbeddingFunction that still calls the
    pre-1.0 openai.Embedding API, which no longer exists. This wraps the
    OpenAI v1+ client instead.
    """

    def __init__(self, model: str):
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model

    def __call__(self, input):
        resp = self._client.embeddings.create(input=input, model=self._model)
        return [d.embedding for d in resp.data]

    def name(self):
        return f"openai-{self._model}"


def get_embedding_function():
    from chromadb.utils import embedding_functions

    if config.EMBED_PROVIDER == "local":
        # No API key needed; downloads a small model on first use.
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.LOCAL_EMBED_MODEL
        )

    return _OpenAIEmbeddingFunction(model=config.EMBED_MODEL)


# --- Generation (the RAG pipeline's answerer) ------------------------------
def generate(prompt: str, system: str = "You are a careful assistant.") -> str:
    if config.LLM_PROVIDER == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=config.GEN_MODEL,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=config.GEN_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


# --- Judge for Ragas -------------------------------------------------------
def get_ragas_judge():
    """Return (llm, embeddings) wrapped for Ragas."""
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import OpenAIEmbeddings

    if config.JUDGE_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=config.JUDGE_MODEL, temperature=0)
    else:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=config.JUDGE_MODEL, temperature=0)

    # Ragas' answer-relevancy metric needs an embedding model regardless of
    # which LLM judges, so embeddings stay on OpenAI here.
    embeddings = OpenAIEmbeddings(model=config.EMBED_MODEL)
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


# --- Judge for DeepEval ----------------------------------------------------
def get_deepeval_model():
    """DeepEval accepts an OpenAI model *name* directly. For Anthropic we wrap
    it in a small custom model. NOTE: newer DeepEval versions pass a `schema`
    arg to generate() for structured output -- if you hit a TypeError, add
    `schema=None` to the signatures below and return schema(**parsed) when set.
    """
    if config.JUDGE_PROVIDER != "anthropic":
        return config.JUDGE_MODEL  # e.g. "gpt-4o-mini"

    from deepeval.models import DeepEvalBaseLLM

    class AnthropicJudge(DeepEvalBaseLLM):
        def __init__(self, model):
            self.model = model

        def load_model(self):
            import anthropic

            return anthropic.Anthropic()

        def generate(self, prompt: str) -> str:
            client = self.load_model()
            msg = client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text

        async def a_generate(self, prompt: str) -> str:
            return self.generate(prompt)

        def get_model_name(self):
            return self.model

    return AnthropicJudge(config.JUDGE_MODEL)
