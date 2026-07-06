import logging
from abc import ABC, abstractmethod
from typing import ClassVar

from sentence_transformers import CrossEncoder

from src.search import SearchHit, Searcher

logger = logging.getLogger(__name__)


class RerankStrategy(ABC):
    """Pluggable scoring for reordering candidate hits.

    A strategy re-scores hits *given* a query; it does not retrieve them.
    Concrete strategies plug into `Reranker` — cross-encoder now; late-interaction
    (ColBERT), listwise LLM, and learned-fusion variants slot in the same way.
    """

    name: ClassVar[str]

    @abstractmethod
    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        """Return hits re-scored and reordered. Does not truncate."""


class CrossEncoderStrategy(RerankStrategy):
    """Full query-document attention scoring via sentence-transformers CrossEncoder.

    One forward pass per (query, hit) pair. Higher precision than any bi-encoder,
    quadratic in candidate count — appropriate for top-K rescoring on small pools
    (≤ 200 candidates). Default checkpoint balances quality and Mac throughput.
    """

    name: ClassVar[str] = "CrossEncoder"
    DEFAULT_MODEL: ClassVar[str] = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        self._model = CrossEncoder(model_id, device=device)

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not hits:
            return []
        pairs = [(query, h.text) for h in hits]
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        rescored = [
            hit.model_copy(update={"score": float(s)})
            for hit, s in zip(hits, scores)
        ]
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored


class Reranker(Searcher):
    """Composite searcher: retrieve K candidates from an inner Searcher, re-score with a strategy.

    Inherits `Searcher` so it drops into `evaluate_retrieval` / `compare_pair`.
    `search(limit=N)` fetches `candidates` from the inner searcher (default 100),
    reranks all of them, and returns the top-N. `candidates` controls the
    oversampling that gives reranking room to improve top-K precision.
    """

    def __init__(
        self,
        retriever: Searcher,
        strategy: RerankStrategy,
        candidates: int = 100,
    ) -> None:
        super().__init__()
        self._retriever = retriever
        self._strategy = strategy
        self.candidates = candidates

    def get_retriever(self) -> Searcher:
        return self._retriever

    def get_strategy(self) -> RerankStrategy:
        return self._strategy

    def search(self, query: str, limit: int = 10) -> list[SearchHit]:
        pool = max(self.candidates, limit)
        hits = self._retriever.search(query, limit=pool)
        reranked = self._strategy.rerank(query, hits)
        return reranked[:limit]
