import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
from fastembed import LateInteractionTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models

from src.dataset import Document
from src.doc_mapping import build_did_to_pids
from src.search import Searcher, SearchHit, dedupe_by_document

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
            for hit, s in zip(hits, scores, strict=True)
        ]
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored


def _maxsim(q_matrix: np.ndarray, d_matrix: np.ndarray) -> float:
    """ColBERT MaxSim: sum over query tokens of max cosine to any doc token.

    Assumes both matrices are L2-normalized (fastembed's ColBERT models
    produce normalized token vectors), so dot product equals cosine similarity.
    """
    sim = q_matrix @ d_matrix.T
    return float(sim.max(axis=1).sum())


class LateInteractionStrategy(RerankStrategy):
    """Local MaxSim reranking with a late-interaction (ColBERT-style) model.

    Encodes the query and each hit's text into per-token vector matrices and
    scores by ColBERT MaxSim. Heavier than a bi-encoder per candidate but
    cheaper than a cross-encoder — appropriate for K ≤ 200. Best when the
    win depends on token-level matching (citations, named entities). Prefer
    `QdrantLateInteractionStrategy` when the ColBERT slot is already indexed
    on the collection — same scoring, no local doc re-embedding.
    """

    name: ClassVar[str] = "LateInteraction"
    DEFAULT_MODEL: ClassVar[str] = "answerdotai/answerai-colbert-small-v1"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        providers: list[str] | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        self._model = LateInteractionTextEmbedding(model_id, providers=providers)

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not hits:
            return []
        q_matrix = np.asarray(
            next(iter(self._model.query_embed(query))), dtype=np.float32
        )
        doc_matrices = self._model.embed(
            [h.text for h in hits], batch_size=self.batch_size
        )
        rescored = [
            hit.model_copy(
                update={"score": _maxsim(q_matrix, np.asarray(dm, dtype=np.float32))}
            )
            for hit, dm in zip(hits, doc_matrices, strict=True)
        ]
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored


class QdrantLateInteractionStrategy(RerankStrategy):
    """Server-side MaxSim reranking via Qdrant Query API filtered to candidates.

    Reuses the collection's pre-indexed late-interaction vectors and Qdrant's
    server-side MaxSim — no local doc re-embedding. The query is embedded
    server-side when the client has cloud inference enabled (sent as
    `models.Document(text=..., model=model_id)`); otherwise it's embedded
    locally with fastembed and sent as a matrix. Preferred over the local
    `LateInteractionStrategy` when the ColBERT slot is already indexed.
    """

    name: ClassVar[str] = "QdrantLateInteraction"

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        vector_name: str,
        model_id: str,
        providers: list[str] | None = None,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.vector_name = vector_name
        self.model_id = model_id
        self.cloud_inference = client.cloud_inference
        self._local_model: LateInteractionTextEmbedding | None = None
        if not self.cloud_inference:
            self._local_model = LateInteractionTextEmbedding(
                model_id, providers=providers
            )

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not hits:
            return []
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=self._query_value(query),
            using=self.vector_name,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchAny(any=[h.doc_id for h in hits]),
                    )
                ]
            ),
            limit=len(hits),
            with_payload=True,
        )
        by_id: dict[str, SearchHit] = {h.doc_id: h for h in hits}
        rescored: list[SearchHit] = []
        for point in result.points:
            payload = point.payload or {}
            doc_id = str(payload.get("doc_id", point.id))
            original = by_id.get(doc_id)
            if original is None:
                continue
            rescored.append(original.model_copy(update={"score": point.score}))
        seen = {h.doc_id for h in rescored}
        rescored.extend(h for h in hits if h.doc_id not in seen)
        return rescored

    def _query_value(self, query: str) -> Any:
        if self.cloud_inference:
            return models.Document(text=query, model=self.model_id)
        assert self._local_model is not None
        matrix = np.asarray(
            next(iter(self._local_model.query_embed(query))), dtype=np.float32
        )
        return matrix.tolist()


class SiblingExpansionStrategy(RerankStrategy):
    """Document-aware reranking for a pre-chunked corpus.

    Expands each candidate hit to every pool passage of its source document
    (its siblings), delegates scoring of the expanded pool to `scorer`, then
    keeps only the best-scoring passage per document. Recovers 'right
    document, wrong slice' retrievals: when the first stage surfaces a sibling
    of the labeled positive, the scorer gets the chance to swap in the exact
    passage. Output is deduplicated by document, so it also diversifies.
    """

    name: ClassVar[str] = "SiblingExpansion"

    def __init__(
        self,
        scorer: RerankStrategy,
        corpus: dict[str, Document],
        pid2did: dict[str, str],
    ) -> None:
        self._scorer = scorer
        self._corpus = corpus
        self._pid2did = pid2did
        self._did_to_pids = build_did_to_pids(corpus, pid2did)

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not hits:
            return []
        expanded: dict[str, SearchHit] = {}
        for hit in hits:
            expanded.setdefault(hit.doc_id, hit)
            for sibling in self._did_to_pids[self._pid2did[hit.doc_id]]:
                if sibling not in expanded:
                    expanded[sibling] = SearchHit(
                        doc_id=sibling,
                        text=self._corpus[sibling].text,
                        score=hit.score,
                        payload={"expanded_from": hit.doc_id},
                    )
        scored = self._scorer.rerank(query, list(expanded.values()))
        return dedupe_by_document(scored, self._pid2did)


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
