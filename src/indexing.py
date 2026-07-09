import logging
import pickle
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, ClassVar, Generic, Literal, TypeVar

import numpy as np
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from pydantic import BaseModel, ConfigDict, Field, model_validator
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    Modifier,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from src.dataset import Document
from src.encoder import OpenRouterEncoder

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

_CACHE_SAVE_EVERY = 2000
_UPSERT_MAX_ATTEMPTS = 5


def _chunked(seq: list[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


class EmbeddingCache:
    """Disk-backed cache of vectors keyed by (model_id, item_id).

    Survives kernel restarts so a stopped/crashed upload doesn't lose work.
    One pickle file per (model_id, kind). Atomic write via tmp-then-rename.
    """

    def __init__(self, cache_dir: str | Path = "./.embedding_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory: dict[tuple[str, str], dict[str, Any]] = {}

    def _path(self, model_id: str, kind: str) -> Path:
        safe = model_id.replace("/", "__")
        return self.cache_dir / f"{safe}.{kind}.pkl"

    def load(self, model_id: str, kind: str) -> dict[str, Any]:
        key = (model_id, kind)
        if key in self._memory:
            return self._memory[key]
        p = self._path(model_id, kind)
        data: dict[str, Any] = {}
        if p.exists():
            with p.open("rb") as f:
                data = pickle.load(f)
        self._memory[key] = data
        return data

    def save(self, model_id: str, kind: str) -> None:
        key = (model_id, kind)
        if key not in self._memory:
            return
        p = self._path(model_id, kind)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(self._memory[key], f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(p)


class EmbeddingConfig(BaseModel):
    """One named vector slot in a Qdrant collection.

    Extend a collection by appending another EmbeddingConfig to the indexer's
    `embeddings` list — the slot's `name` becomes the named-vector key on every
    PointStruct.
    """

    model_config = ConfigDict(protected_namespaces=())

    name: str
    model_id: str
    kind: Literal["dense", "sparse", "late_interaction"]
    backend: Literal["fastembed", "sentence-transformers", "openrouter"] = "fastembed"
    """Which embedding library to load `model_id` with. `sentence-transformers`
    unlocks MPS/CUDA acceleration for dense models; `fastembed` is required for
    sparse (BM25/SPLADE/miniCOIL); `openrouter` embeds client-side via the
    OpenRouter API (even when the Qdrant client runs with cloud_inference) —
    the api key comes from doc_options['openrouter-api-key'] or the
    OPENROUTER_API_KEY env var, remaining doc_options entries (e.g.
    `dimensions`) are forwarded in the request body."""
    size: int | None = None
    distance: Distance = Distance.COSINE
    providers: list[str] | None = None
    """fastembed ONNX Runtime execution providers. Ignored when
    backend='sentence-transformers'."""
    parallel: int | None = None
    """fastembed multi-process workers for CPU inference. None = single process.
    Set to number of CPU cores - 1 for ~2-4x speedup on CPU. Leave None when
    using GPU providers. Ignored when backend='sentence-transformers'."""
    device: str | None = None
    """sentence-transformers device ('mps', 'cuda', 'cpu'). None auto-detects.
    Ignored when backend='fastembed'."""
    query_prompt: str | None = None
    """Prefix prepended to queries at search time. Required for bge dense models
    under sentence-transformers to match fastembed's built-in query formatting:
    `'Represent this sentence for searching relevant passages: '`. Skipping it
    silently drops retrieval quality."""
    modifier: Modifier | None = None
    """Server-side score adjustment for sparse vectors. Set `Modifier.IDF` for
    BM25/miniCOIL so Qdrant applies IDF using collection-wide term stats;
    without it, sparse scoring falls back to raw TF and quality drops."""
    hnsw_m: int | None = None
    """Per-slot HNSW `m` override (dense/late-interaction only). Set 0 for
    slots used exclusively to rescore prefetched candidates (e.g. a ColBERT
    rerank slot) — no graph is built, saving hours of background indexing on
    multivectors. None keeps the server default."""
    
    doc_options: dict[str, Any] | None = Field(
        None,
        description="Additional options for the document configuration",
        repr=False,
        exclude=True,
    )
    """May carry provider API keys — kept out of repr and model_dump so it
    never leaks into notebook outputs or serialized configs."""

    @model_validator(mode="after")
    def _check_backend(self) -> "EmbeddingConfig":
        if self.backend != "fastembed" and self.kind != "dense":
            raise ValueError(
                f"{self.backend} backend supports dense embeddings only; "
                "keep sparse/late-interaction slots on fastembed."
            )
        return self

    def openrouter_encoder_kwargs(self) -> dict[str, Any]:
        """Split doc_options into the encoder's api_key + request options."""
        opts = dict(self.doc_options or {})
        api_key = opts.pop("openrouter-api-key", None)
        return {"model": self.model_id, "api_key": api_key, "options": opts}


class BaseIndexer(ABC, Generic[T]):
    """Manage a Qdrant collection of items of type T.

    Subclasses bind T via `item_type` and implement item_id/text/payload. The
    base class owns collection creation, batched embedding across N vector
    slots, and upload.
    """

    item_type: ClassVar[type[BaseModel]]

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embeddings: list[EmbeddingConfig],
        cache: EmbeddingCache | None = None,
    ) -> None:
        if not embeddings:
            raise ValueError("Indexer requires at least one EmbeddingConfig.")
        self.client = client
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.cache = cache
        self.cloud_inference = client.cloud_inference
        self._dense_models: dict[str, TextEmbedding] = {}
        self._late_interaction_models: dict[str, LateInteractionTextEmbedding] = {}
        self._sparse_models: dict[str, SparseTextEmbedding] = {}
        self._st_models: dict[str, SentenceTransformer] = {}
        self._openrouter_models: dict[str, OpenRouterEncoder] = {}

    @abstractmethod
    def item_id(self, item: T) -> int | str: ...

    @abstractmethod
    def item_text(self, item: T) -> str: ...

    @abstractmethod
    def item_payload(self, item: T) -> dict[str, Any]: ...

    def ensure_collection(self, recreate: bool = False) -> None:
        """Create the collection with the configured vector slots.

        If the collection already exists and `recreate` is False, no-op. If
        True, drop and recreate so schema changes (new vector slot) take effect.
        """
        exists = self.client.collection_exists(self.collection_name)
        if exists and not recreate:
            return
        if exists:
            self.client.delete_collection(self.collection_name)

        dense_config: dict[str, VectorParams] = {}
        for cfg in self.embeddings:
            hnsw = HnswConfigDiff(m=cfg.hnsw_m) if cfg.hnsw_m is not None else None
            if cfg.kind == "dense":
                dense_config[cfg.name] = VectorParams(
                    size=self._dense_size(cfg),
                    distance=cfg.distance,
                    hnsw_config=hnsw,
                )
            elif cfg.kind == "late_interaction":
                dense_config[cfg.name] = VectorParams(
                    size=self._dense_size(cfg),
                    distance=cfg.distance,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM,
                    ),
                    hnsw_config=hnsw,
                )
        sparse_config = {
            cfg.name: SparseVectorParams(modifier=cfg.modifier)
            for cfg in self.embeddings
            if cfg.kind == "sparse"
        }
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=dense_config or None,
            sparse_vectors_config=sparse_config or None,
        )

    def upload(
        self,
        items: Sequence[T],
        batch_size: int = 64,
        parallel: int = 1,
        skip_existing: bool = False,
    ) -> None:
        """Embed `items` for every configured slot and upload as PointStructs.

        `parallel` > 1 keeps that many upsert requests in flight — the main
        lever under `cloud_inference`, where server-side embedding dominates
        request latency and sequential batches leave the connection idle.
        `skip_existing` drops items whose id is already in the collection, so
        an interrupted bulk upload resumes without re-paying inference.

        Raises TypeError if items aren't `item_type` — guards against using the
        wrong indexer subclass for the data.
        """
        if not items:
            return
        if not isinstance(items[0], self.item_type):
            raise TypeError(
                f"{type(self).__name__} expects {self.item_type.__name__}, "
                f"got {type(items[0]).__name__}"
            )

        if skip_existing:
            existing = self._existing_ids([self.item_id(i) for i in items])
            if existing:
                logger.info(
                    f"Skipping {len(existing)} of {len(items)} items already "
                    f"in {self.collection_name!r}"
                )
            items = [i for i in items if self.item_id(i) not in existing]
            if not items:
                return

        texts = [self.item_text(i) for i in items]
        ids = [self.item_id(i) for i in items]
        payloads = [self.item_payload(i) for i in items]

        vectors: list[dict[str, Any]] = [{} for _ in items]
        for cfg in self.embeddings:
            for i, vec in enumerate(self._vectors_for(cfg, ids, texts, batch_size)):
                vectors[i][cfg.name] = vec

        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(items))
        ]
        chunks = list(_chunked(points, batch_size))
        if parallel <= 1:
            for chunk in tqdm(chunks, desc="upsert"):
                self._upsert(chunk)
            return
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = [executor.submit(self._upsert, chunk) for chunk in chunks]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="upsert"
            ):
                future.result()

    def _upsert(self, points: list[PointStruct]) -> None:
        # Under cloud inference the server proxies to external embedding
        # providers, which intermittently return 5xx — retry those, fail
        # fast on 4xx (bad request won't heal).
        for attempt in range(1, _UPSERT_MAX_ATTEMPTS + 1):
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )
                return
            except UnexpectedResponse as e:
                status = e.status_code or 0
                if status < 500 or attempt == _UPSERT_MAX_ATTEMPTS:
                    raise
                delay = 2**attempt
                logger.warning(
                    f"Upsert got {status} (attempt {attempt}/"
                    f"{_UPSERT_MAX_ATTEMPTS}), retrying in {delay}s: "
                    f"{e.content[:200] if e.content else ''}"
                )
                time.sleep(delay)

    def _existing_ids(self, ids: list[int | str]) -> set[int | str]:
        """Ids from `ids` that are already stored in the collection."""
        existing: set[int | str] = set()
        for chunk in _chunked(ids, 1000):
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=chunk,
                with_payload=False,
                with_vectors=False,
            )
            existing.update(record.id for record in records)
        return existing

    def _vectors_for(
        self,
        cfg: EmbeddingConfig,
        ids: list[int | str],
        texts: list[str],
        batch_size: int,
    ) -> list[Any]:
        """Return vectors aligned with `ids`/`texts`, hitting cache where possible.

        Under `cloud_inference=True`, texts are wrapped as `Document` payloads —
        Qdrant embeds server-side, so no local model runs and the on-disk cache
        does not apply. Exception: `backend='openrouter'` slots always embed
        client-side (direct API calls), so they stay cacheable and skip the
        slow synchronous inference proxy.
        """
        if self.cloud_inference and cfg.backend != "openrouter":
            return [
                models.Document(text=t, model=cfg.model_id, options=cfg.doc_options)
                for t in texts
            ]
        if self.cache is None:
            return self._embed(cfg, texts, batch_size)

        store = self.cache
        cache = store.load(cfg.model_id, cfg.kind)

        missing_idx = [i for i, id_ in enumerate(ids) if str(id_) not in cache]
        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            try:
                new_vecs = self._embed(cfg, missing_texts, batch_size)
                for pos, original_i in enumerate(missing_idx):
                    cache[str(ids[original_i])] = new_vecs[pos]
                    if (pos + 1) % _CACHE_SAVE_EVERY == 0:
                        store.save(cfg.model_id, cfg.kind)
            finally:
                # flush whatever we got, even on Ctrl-C / OOM mid-embedding
                store.save(cfg.model_id, cfg.kind)

        return [cache[str(id_)] for id_ in ids]

    def _embed(
        self,
        cfg: EmbeddingConfig,
        texts: list[str],
        batch_size: int,
    ) -> list[Any]:
        if cfg.kind == "late_interaction":
            if cfg.backend == 'sentence-transformers':
                raise NotImplementedError("There is no currently supported implementation for the late interaction models with 'sentence-transformers'")
            return self._embed_late_fastembed(cfg, texts, batch_size)
        if cfg.kind == "dense":
            if cfg.backend == "sentence-transformers":
                return self._embed_dense_st(cfg, texts, batch_size)
            if cfg.backend == "openrouter":
                return self._openrouter(cfg).encode(
                    texts, batch_size=batch_size, parallel=cfg.parallel or 8
                )
            return self._embed_dense_fastembed(cfg, texts, batch_size)

        model = self._sparse(cfg)
        stream = tqdm(
            model.embed(texts, batch_size=batch_size, parallel=cfg.parallel),
            total=len(texts),
            desc=f"embed:{cfg.name}",
        )
        return [
            SparseVector(
                indices=np.asarray(s.indices, dtype=np.int64).tolist(),
                values=np.asarray(s.values, dtype=np.float32).tolist(),
            )
            for s in stream
        ]

    def _embed_dense_fastembed(
        self, cfg: EmbeddingConfig, texts: list[str], batch_size: int
    ) -> list[np.ndarray]:
        model = self._dense(cfg)
        stream = tqdm(
            model.embed(texts, batch_size=batch_size, parallel=cfg.parallel),
            total=len(texts),
            desc=f"embed:{cfg.name}",
        )
        return [np.asarray(v, dtype=np.float32) for v in stream]

    def _embed_late_fastembed(
        self, cfg: EmbeddingConfig, texts: list[str], batch_size: int
    ) -> list[np.ndarray]:
        model =  self._late_interaction(cfg)
        stream = tqdm(
            model.embed(texts, batch_size=batch_size, parallel=cfg.parallel),
            total=len(texts),
            desc=f"late_interaction:{cfg.name}",
        )
        return [np.asarray(v, dtype=np.float32) for v in stream]

    def _embed_dense_st(
        self, cfg: EmbeddingConfig, texts: list[str], batch_size: int
    ) -> list[np.ndarray]:
        model = self._st(cfg)
        matrix = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=cfg.distance == Distance.COSINE,
        )
        return [np.asarray(v, dtype=np.float32) for v in matrix]

    def _dense(self, cfg: EmbeddingConfig) -> TextEmbedding:
        if cfg.model_id not in self._dense_models:
            self._dense_models[cfg.model_id] = TextEmbedding(
                cfg.model_id, providers=cfg.providers
            )
        return self._dense_models[cfg.model_id]

    def _late_interaction(self, cfg: EmbeddingConfig) -> LateInteractionTextEmbedding:
        if cfg.model_id not in self._dense_models:
            self._late_interaction_models[cfg.model_id] = LateInteractionTextEmbedding(
                cfg.model_id, providers=cfg.providers
            )
        return self._late_interaction_models[cfg.model_id]

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

    def _openrouter(self, cfg: EmbeddingConfig) -> OpenRouterEncoder:
        if cfg.model_id not in self._openrouter_models:
            self._openrouter_models[cfg.model_id] = OpenRouterEncoder(
                **cfg.openrouter_encoder_kwargs()
            )
        return self._openrouter_models[cfg.model_id]

    @staticmethod
    def _dense_size(cfg: EmbeddingConfig) -> int:
        if cfg.size is None:
            raise ValueError(f"Dense embedding {cfg.name!r} requires `size`.")
        return cfg.size


class DocumentIndexer(BaseIndexer[Document]):
    """Index Document items (doc_id, text) — the CLERC corpus."""

    item_type = Document

    def item_id(self, item: Document) -> int:
        return int(item.doc_id)

    def item_text(self, item: Document) -> str:
        return item.text

    def item_payload(self, item: Document) -> dict[str, Any]:
        return {"doc_id": item.doc_id, "text": item.text}
