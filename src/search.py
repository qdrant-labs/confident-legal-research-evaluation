import logging
from typing import Any

from fastembed import TextEmbedding
from pydantic import BaseModel
from qdrant_client import QdrantClient

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


class DenseSearcher:
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
        self.client = client
        self.collection_name = collection_name
        self.embedding = embedding
        self._model: TextEmbedding | None = None

    def search(self, query: str, limit: int = 10) -> list[SearchHit]:
        """Return the top-`limit` hits ranked by similarity to `query`."""
        vector = next(iter(self._get_model().query_embed(query)))
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=vector.tolist(),
            using=self.embedding.name,
            limit=limit,
            with_payload=True,
        )

        hits: list[SearchHit] = []
        for point in result.points:
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

    def _get_model(self) -> TextEmbedding:
        if self._model is None:
            self._model = TextEmbedding(
                self.embedding.model_id,
                providers=self.embedding.providers,
            )
        return self._model
