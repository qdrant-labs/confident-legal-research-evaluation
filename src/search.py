import logging
from abc import ABC, abstractmethod
from typing import Any

from fastembed import SparseTextEmbedding, TextEmbedding
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    Prefetch,
    ScoredPoint,
    SparseVector,
)
from sentence_transformers import SentenceTransformer

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


class Searcher(ABC):
    """Ranked-retrieval interface over a Qdrant collection.

    Subclasses choose the query strategy (single dense slot, fused multi-slot,
    reranked, ...); consumers depend only on `search()` returning SearchHits.
    Owns per-backend model caches so subclasses just call the query helpers.
    """

    def __init__(self) -> None:
        self._dense_models: dict[str, TextEmbedding] = {}
        self._sparse_models: dict[str, SparseTextEmbedding] = {}
        self._st_models: dict[str, SentenceTransformer] = {}

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> list[SearchHit]:
        """Return the top-`limit` hits ranked by relevance to `query`."""

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

    def _st(self, cfg: EmbeddingConfig) -> SentenceTransformer:
        if cfg.model_id not in self._st_models:
            self._st_models[cfg.model_id] = SentenceTransformer(
                cfg.model_id, device=cfg.device
            )
        return self._st_models[cfg.model_id]


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

    def search(self, query: str, limit: int = 10) -> list[SearchHit]:
        vector = self._embed_dense_query(self.embedding, query)
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
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
        if cfg.kind == "dense":
            vector = self._embed_dense_query(cfg, query)
            return Prefetch(query=vector, using=cfg.name, limit=limit)
        sparse = self._embed_sparse_query(cfg, query)
        return Prefetch(query=sparse, using=cfg.name, limit=limit)
