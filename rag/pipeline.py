"""A minimal but real Chroma-backed RAG pipeline.

This is the "system under test". The eval suite holds everything here fixed
except the one variable being tested (model, prompt, top_k, embeddings...),
which is what makes a run a clean regression check.
"""
import chromadb

import config
from llm import generate, get_embedding_function
from rag.corpus import DOCUMENTS

# The instruction prompt is part of the system under test -- version it, and
# expect prompt edits to move the faithfulness and answer-relevancy metrics.
SYSTEM = "You are a support assistant. Answer using ONLY the provided context."

TEMPLATE = """Context:
{context}

Question: {question}

Answer using only the context above. If the context does not contain the
answer, reply exactly: "I don't know based on the available information."
"""


class RagPipeline:
    def __init__(self, persist_dir=None, collection=None, k=None):
        self.client = chromadb.PersistentClient(path=persist_dir or config.CHROMA_DIR)
        self.embedding_function = get_embedding_function()
        self.collection = self.client.get_or_create_collection(
            name=collection or config.COLLECTION,
            embedding_function=self.embedding_function,
        )
        self.k = k or config.TOP_K

    def ingest(self, documents=DOCUMENTS):
        """Idempotent: upsert so re-running doesn't error on duplicate ids."""
        self.collection.upsert(
            ids=[d["id"] for d in documents],
            documents=[d["text"] for d in documents],
            metadatas=[{"source": d["source"]} for d in documents],
        )
        return self.collection.count()

    def retrieve(self, question: str, k: int | None = None) -> list[str]:
        result = self.collection.query(
            query_texts=[question], n_results=k or self.k
        )
        return result["documents"][0]

    def answer(self, question: str, k: int | None = None) -> tuple[str, list[str]]:
        """Return (answer, retrieved_contexts) -- exactly the tuple the
        eval metrics consume."""
        contexts = self.retrieve(question, k)
        context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        prompt = TEMPLATE.format(context=context_block, question=question)
        return generate(prompt, system=SYSTEM), contexts


if __name__ == "__main__":
    pipe = RagPipeline()
    print(f"Ingested {pipe.ingest()} documents.")
    ans, ctx = pipe.answer("How much does the Pro plan cost?")
    print("\nAnswer:", ans)
    print("\nRetrieved:", ctx)
