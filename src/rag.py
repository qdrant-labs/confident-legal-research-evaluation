import logging

import instructor
import litellm
from pydantic import BaseModel, Field

from src.search import DenseSearcher, SearchHit

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"
DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_RETRIES = 2

DEFAULT_SYSTEM_PROMPT = (
    "You are a research assistant for junior associates at a law firm. "
    "Answer the user's question using ONLY the case excerpts provided in the "
    "user message. Every factual claim must cite the doc_id of the supporting "
    "excerpt. If the excerpts do not contain enough information to answer the "
    "question, say so explicitly — never speculate and never rely on outside "
    "knowledge."
)


class Citation(BaseModel):
    """A single citation grounding a claim in the answer."""

    doc_id: str = Field(description="doc_id of the cited excerpt.")
    quote: str = Field(
        description=(
            "The specific sentence(s) from the excerpt that support the claim."
        )
    )


class _AnswerPayload(BaseModel):
    """Final answer to the user's question, with citations grounding each claim."""

    answer: str = Field(
        description=(
            "The answer, grounded in the provided excerpts. Plain prose, "
            "no markdown."
        )
    )
    citations: list[Citation] = Field(
        description=(
            "Source citations. Every factual claim in `answer` must be "
            "backed by at least one citation."
        )
    )


class RAGAnswer(BaseModel):
    """Final RAG output: synthesized answer, citations, and retrieved context."""

    answer: str
    citations: list[Citation]
    retrieved: list[SearchHit]

    def __str__(self) -> str:
        lines = [self.answer.strip(), ""]
        if self.citations:
            lines.append("Citations:")
            for c in self.citations:
                quote = c.quote if len(c.quote) <= 160 else c.quote[:157] + "..."
                lines.append(f"  [{c.doc_id}] {quote}")
            lines.append("")
        lines.append(f"Retrieved {len(self.retrieved)} hit(s).")
        return "\n".join(lines)


class LegalAssistant:
    """Retrieval-augmented Q&A for junior law firm associates.

    Pipeline: dense search via `DenseSearcher` → synthesize a cited answer
    via instructor + litellm, with response shape enforced by `_AnswerPayload`.
    """

    def __init__(
        self,
        searcher: DenseSearcher,
        model: str = DEFAULT_MODEL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        top_k: int = DEFAULT_TOP_K,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.searcher = searcher
        self.model = model
        self.system_prompt = system_prompt
        self.top_k = top_k
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        # Tool-call mode: structured output via function-calling, not JSON-mode.
        self._client = instructor.from_litellm(
            litellm.completion, mode=instructor.Mode.TOOLS
        )

    def ask(self, question: str) -> RAGAnswer:
        """Retrieve relevant excerpts and synthesize a cited answer."""
        hits = self.searcher.search(question, limit=self.top_k)
        if not hits:
            return RAGAnswer(
                answer=(
                    "No relevant case excerpts were found for this question. "
                    "Try rephrasing or broadening the query."
                ),
                citations=[],
                retrieved=[],
            )

        payload: _AnswerPayload = self._client.chat.completions.create(
            model=self.model,
            response_model=_AnswerPayload,
            max_retries=self.max_retries,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": self._render_user_message(question, hits),
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return RAGAnswer(
            answer=payload.answer,
            citations=payload.citations,
            retrieved=hits,
        )

    @staticmethod
    def _render_user_message(question: str, hits: list[SearchHit]) -> str:
        excerpts = "\n\n".join(
            f"[doc_id: {hit.doc_id}]\n{hit.text}" for hit in hits
        )
        return (
            f"Question: {question}\n\n"
            f"Case excerpts (retrieved by relevance):\n{excerpts}\n\n"
            "Answer the question using only the excerpts above."
        )
