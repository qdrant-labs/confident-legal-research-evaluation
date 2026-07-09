import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    Prefetch,
    ScoredPoint,
    SparseVector,
)
from sentence_transformers import SentenceTransformer

from src.encoder import OpenRouterEncoder
from src.indexing import EmbeddingConfig

logger = logging.getLogger(__name__)


class SearchHit(BaseModel):
    """A single retrieval result.

    `doc_id` and `text` are lifted from the payload for ergonomics; the full
    payload is preserved in `payload` for downstream consumers.
    """

    doc_id: str
    text: str
    score: float
    payload: dict[str, Any]


def dedupe_by_document(
    hits: list[SearchHit],
    pid2did: dict[str, str],
    limit: int | None = None,
) -> list[SearchHit]:
    """Keep the first (best-scored) hit per source document, preserving order."""
    seen_dids: set[str] = set()
    unique: list[SearchHit] = []
    for hit in hits:
        did = pid2did[hit.doc_id]
        if did in seen_dids:
            continue
        seen_dids.add(did)
        unique.append(hit)
        if limit is not None and len(unique) == limit:
            break
    return unique


class Searcher(ABC):
    """Ranked-retrieval interface over a Qdrant collection.

    Subclasses choose the query strategy (single dense slot, fused multi-slot,
    reranked, ...); consumers depend only on `search()` returning SearchHits.
    Owns per-backend model caches so subclasses just call the query helpers.
    """

    def __init__(self) -> None:
        self.cloud_inference = False
        self._dense_models: dict[str, TextEmbedding] = {}
        self._sparse_models: dict[str, SparseTextEmbedding] = {}
        self._late_interaction_models: dict[str, LateInteractionTextEmbedding] = {}
        self._st_models: dict[str, SentenceTransformer] = {}
        self._openrouter_models: dict[str, OpenRouterEncoder] = {}
        self._query_vector_cache: dict[tuple[str, str], list[float]] = {}

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> list[SearchHit]:
        """Return the top-`limit` hits ranked by relevance to `query`."""

    def warm_up(
        self,
        queries: Sequence[str],
        batch_size: int = 64,
        parallel: int = 8,
    ) -> None:
        """Batch-embed `queries` for every openrouter dense slot ahead of time.

        Fills the query-vector cache so subsequent `search()` calls skip the
        per-query API round trip — turns N sequential requests during an eval
        loop into N/batch_size concurrent ones up front. Queries not warmed up
        still embed live on first use.
        """
        for cfg in self._warm_up_configs():
            if cfg.backend != "openrouter" or cfg.kind != "dense":
                continue
            texts = [(cfg.query_prompt or "") + q for q in queries]
            missing = [
                t
                for t in dict.fromkeys(texts)
                if (cfg.model_id, t) not in self._query_vector_cache
            ]
            if not missing:
                continue
            vectors = self._openrouter(cfg).encode(
                missing, batch_size=batch_size, parallel=parallel
            )
            for text, vec in zip(missing, vectors):
                self._query_vector_cache[(cfg.model_id, text)] = vec.tolist()

    def _warm_up_configs(self) -> list[EmbeddingConfig]:
        """Configs whose query embeddings `warm_up` should precompute."""
        return []

    def _query_value(self, cfg: EmbeddingConfig, query: str) -> Any:
        """Query representation for one slot: a `models.Document` embedded
        server-side under cloud inference, a locally embedded vector otherwise.
        `backend='openrouter'` slots always embed client-side, mirroring the
        indexing path."""
        if self.cloud_inference and cfg.backend != "openrouter":
            return models.Document(
                text=query, model=cfg.model_id, options=cfg.doc_options
            )
        if cfg.kind == "dense":
            return self._embed_dense_query(cfg, query)
        if cfg.kind == "sparse":
            return self._embed_sparse_query(cfg, query)
        if cfg.kind == "late_interaction":
            return self._embed_late_interaction_query(cfg, query)
        raise ValueError(f"Unknown embedding kind: {cfg.kind!r}")

    @staticmethod
    def _hits_from_points(points: list[ScoredPoint]) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for point in points:
            payload = point.payload or {}
            hits.append(
                SearchHit(
                    doc_id=str(payload.get("doc_id", point.id)),
                    text=payload.get("text", ""),
                    score=point.score,
                    payload=payload,
                )
            )
        return hits

    def _embed_dense_query(self, cfg: EmbeddingConfig, query: str) -> list[float]:
        if cfg.backend == "openrouter":
            text = (cfg.query_prompt or "") + query
            key = (cfg.model_id, text)
            if key not in self._query_vector_cache:
                encoder = self._openrouter(cfg)
                vec = encoder.encode([text], show_progress_bar=False)[0]
                self._query_vector_cache[key] = vec.tolist()
            return self._query_vector_cache[key]
        if cfg.backend == "sentence-transformers":
            model = self._st(cfg)
            text = (cfg.query_prompt or "") + query
            vec = model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=cfg.distance == Distance.COSINE,
            )
            return vec.tolist()
        model = self._dense(cfg)
        vec = next(iter(model.query_embed(query)))
        return vec.tolist()

    def _embed_sparse_query(
        self, cfg: EmbeddingConfig, query: str
    ) -> SparseVector:
        model = self._sparse(cfg)
        emb = next(iter(model.query_embed(query)))
        return SparseVector(
            indices=emb.indices.tolist(),
            values=emb.values.tolist(),
        )

    def _embed_late_interaction_query(
        self, cfg: EmbeddingConfig, query: str
    ) -> list[list[float]]:
        model = self._late_interaction(cfg)
        vec = next(iter(model.query_embed(query)))
        return np.asarray(vec, dtype=np.float32).tolist()

    def _dense(self, cfg: EmbeddingConfig) -> TextEmbedding:
        if cfg.model_id not in self._dense_models:
            self._dense_models[cfg.model_id] = TextEmbedding(
                cfg.model_id, providers=cfg.providers
            )
        return self._dense_models[cfg.model_id]

    def _sparse(self, cfg: EmbeddingConfig) -> SparseTextEmbedding:
        if cfg.model_id not in self._sparse_models:
            self._sparse_models[cfg.model_id] = SparseTextEmbedding(
                cfg.model_id, providers=cfg.providers
            )
        return self._sparse_models[cfg.model_id]

    def _late_interaction(self, cfg: EmbeddingConfig) -> LateInteractionTextEmbedding:
        if cfg.model_id not in self._late_interaction_models:
            self._late_interaction_models[cfg.model_id] = LateInteractionTextEmbedding(
                cfg.model_id, providers=cfg.providers
            )
        return self._late_interaction_models[cfg.model_id]

    def _st(self, cfg: EmbeddingConfig) -> SentenceTransformer:
        if cfg.model_id not in self._st_models:
            self._st_models[cfg.model_id] = SentenceTransformer(
                cfg.model_id, device=cfg.device
            )
        return self._st_models[cfg.model_id]

    def _openrouter(self, cfg: EmbeddingConfig) -> OpenRouterEncoder:
        if cfg.model_id not in self._openrouter_models:
            self._openrouter_models[cfg.model_id] = OpenRouterEncoder(
                **cfg.openrouter_encoder_kwargs()
            )
        return self._openrouter_models[cfg.model_id]


class UniqueDocSearcher(Searcher):
    """Document-level diversification over a passage-level retriever.

    Oversamples the inner searcher and keeps only the best-ranked passage per
    source document, so top-`limit` holds `limit` distinct documents instead of
    near-duplicate slices of the same one — the classic mitigation for a
    pre-chunked corpus. Document identity comes from `pid2did`, so no
    reindexing or payload changes are required.
    """

    def __init__(
        self,
        retriever: Searcher,
        pid2did: dict[str, str],
        oversample: int = 5,
    ) -> None:
        super().__init__()
        self._retriever = retriever
        self._pid2did = pid2did
        self.oversample = oversample

    def search(self, query: str, limit: int = 10) -> list[SearchHit]:
        hits = self._retriever.search(query, limit=limit * self.oversample)
        return dedupe_by_document(hits, self._pid2did, limit=limit)

    def warm_up(
        self,
        queries: Sequence[str],
        batch_size: int = 64,
        parallel: int = 8,
    ) -> None:
        # the inner retriever embeds the queries, so its cache must be filled
        self._retriever.warm_up(queries, batch_size=batch_size, parallel=parallel)


class DenseSearcher(Searcher):
    """Dense vector search against a named slot of a Qdrant collection.

    Embeds the query with the same EmbeddingConfig used at index time and
    queries the matching vector slot via `client.query_points(..., using=...)`.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding: EmbeddingConfig,
    ) -> None:
        if embedding.kind != "dense":
            raise ValueError(
                f"DenseSearcher requires a dense EmbeddingConfig; "
                f"got kind={embedding.kind!r}."
            )
        super().__init__()
        self.client = client
        self.collection_name = collection_name
        self.embedding = embedding
        self.cloud_inference = client.cloud_inference

    def search(self, query: str, limit: int = 10) -> list[SearchHit]:
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=self._query_value(self.embedding, query),
            using=self.embedding.name,
            limit=limit,
            with_payload=True,
        )
        return self._hits_from_points(result.points)

    def _warm_up_configs(self) -> list[EmbeddingConfig]:
        return [self.embedding]

class SparseSearcher(Searcher):
    """Sparse vector search against a named slot of a Qdrant collection.

    Embeds the query with the same EmbeddingConfig used at index time and
    queries the matching vector slot via `client.query_points(..., using=...)`.
    """
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding: EmbeddingConfig,
    ) -> None:
        if embedding.kind != "sparse":
            raise ValueError(
                f"SparseSearch requires a sparse vector in EmbeddingConfig; "
                f"got kind={embedding.kind!r}."
            )
        super().__init__()
        self.client = client
        self.collection_name = collection_name
        self.embedding = embedding
        self.cloud_inference = client.cloud_inference

    def search(self, query: str, limit: int = 10) -> list[SearchHit]:
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=self._query_value(self.embedding, query),
            using=self.embedding.name,
            limit=limit,
            with_payload=True,
        )
        return self._hits_from_points(result.points)

class HybridSearcher(Searcher):
    """Fused search across multiple named-vector slots via Qdrant's Query API.

    Issues one `prefetch` per `EmbeddingConfig` (dense and/or sparse) in a single
    request and fuses the ranked lists server-side. RRF is the default fusion —
    rank-based, robust when prefetch score scales are not comparable (BM25 vs
    cosine). Prefetch depth is `limit * prefetch_multiplier`; oversampling the
    candidate pool is what gives fusion room to improve top-K precision.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embeddings: list[EmbeddingConfig],
        fusion: Fusion = Fusion.RRF,
        prefetch_multiplier: int = 5,
    ) -> None:
        if len(embeddings) < 2:
            raise ValueError(
                "HybridSearcher expects at least two EmbeddingConfigs; "
                "use DenseSearcher for a single-slot query."
            )
        super().__init__()
        self.client = client
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.fusion = fusion
        self.prefetch_multiplier = prefetch_multiplier
        self.cloud_inference = client.cloud_inference

    def search(self, query: str, limit: int = 10) -> list[SearchHit]:
        prefetch_limit = limit * self.prefetch_multiplier
        prefetches = [
            self._build_prefetch(cfg, query, prefetch_limit)
            for cfg in self.embeddings
        ]
        result = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetches,
            query=FusionQuery(fusion=self.fusion),
            limit=limit,
            with_payload=True,
        )
        return self._hits_from_points(result.points)

    def _build_prefetch(
        self, cfg: EmbeddingConfig, query: str, limit: int
    ) -> Prefetch:
        return Prefetch(
            query=self._query_value(cfg, query),
            using=cfg.name,
            limit=limit,
        )

    def _warm_up_configs(self) -> list[EmbeddingConfig]:
        return self.embeddings


class HybridRerankSearcher(Searcher):
    """Server-side rerank: fuse candidates from multiple slots, rescore with a final slot.

    One Qdrant Query API call: an inner nested `Prefetch(FusionQuery(RRF))` merges
    the retrieval slots (typically dense + sparse), and the outer `query` uses the
    rerank slot (typically late-interaction / multi-vector) to reorder the fused
    pool. Embedding + fusion + rerank all happen server-side.

    When `client.cloud_inference` is True, queries are sent as
    `models.Document(text=..., model=cfg.model_id)` and embedded server-side —
    required for late-interaction, since no local ColBERT path is wired.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        prefetch: list[EmbeddingConfig],
        rerank: EmbeddingConfig,
        fusion: Fusion = Fusion.RRF,
        prefetch_multiplier: int = 10,
    ) -> None:
        if not prefetch:
            raise ValueError(
                "HybridRerankSearcher requires at least one prefetch EmbeddingConfig."
            )
        super().__init__()
        self.client = client
        self.collection_name = collection_name
        self.prefetch = prefetch
        self.rerank = rerank
        self.fusion = fusion
        self.prefetch_multiplier = prefetch_multiplier
        self.cloud_inference = client.cloud_inference

    def search(self, query: str, limit: int = 10) -> list[SearchHit]:
        pool = limit * self.prefetch_multiplier
        fused = Prefetch(
            prefetch=[self._as_prefetch(c, query, pool) for c in self.prefetch],
            query=FusionQuery(fusion=self.fusion),
            limit=pool,
        )
        result = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[fused],
            query=self._query_value(self.rerank, query),
            using=self.rerank.name,
            limit=limit,
            with_payload=True,
        )
        return self._hits_from_points(result.points)

    def _as_prefetch(
        self, cfg: EmbeddingConfig, query: str, limit: int
    ) -> Prefetch:
        return Prefetch(
            query=self._query_value(cfg, query),
            using=cfg.name,
            limit=limit,
        )

    def _warm_up_configs(self) -> list[EmbeddingConfig]:
        return [*self.prefetch, self.rerank]
