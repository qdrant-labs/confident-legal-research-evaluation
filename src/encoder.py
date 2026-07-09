import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx
import numpy as np
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

_API_URL = "https://openrouter.ai/api/v1/embeddings"
_RETRYABLE = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 5


def post_with_retry(
    client: httpx.Client,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
) -> httpx.Response:
    """POST with exponential backoff on transient OpenRouter errors."""
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        response = client.post(url, json=body, headers=headers)
        if response.status_code in _RETRYABLE and attempt < _MAX_ATTEMPTS:
            delay = 2**attempt
            logger.warning(
                f"OpenRouter returned {response.status_code} (attempt "
                f"{attempt}/{_MAX_ATTEMPTS}), retrying in {delay}s"
            )
            time.sleep(delay)
            continue
        response.raise_for_status()
        return response
    raise RuntimeError("unreachable")


class OpenRouterEncoder:
    """SentenceTransformer-style encoder backed by the OpenRouter embeddings API.

    `encode` chunks texts into batches, sends each batch as a single API
    request (the API accepts a list of inputs), keeps `parallel` requests in
    flight, and reassembles embeddings in the original text order.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        options: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model.removeprefix("openrouter/")
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "OpenRouter API key missing: pass `api_key` or set the "
                "OPENROUTER_API_KEY environment variable."
            )
        self.options = options or {}
        self._headers = {"Authorization": f"Bearer {key}"}
        self._client = httpx.Client(timeout=timeout)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        parallel: int = 8,
        show_progress_bar: bool = True,
    ) -> list[np.ndarray]:
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        results: list[list[np.ndarray] | None] = [None] * len(batches)
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(self._embed_batch, batch): i
                for i, batch in enumerate(batches)
            }
            iterator = as_completed(futures)
            if show_progress_bar:
                iterator = tqdm(
                    iterator, total=len(batches), desc=f"openrouter:{self.model}"
                )
            for future in iterator:
                results[futures[future]] = future.result()
        return [vec for batch in results if batch is not None for vec in batch]

    def _embed_batch(self, batch: list[str]) -> list[np.ndarray]:
        body = {"model": self.model, "input": batch, **self.options}
        response = post_with_retry(self._client, _API_URL, body, self._headers)
        data = sorted(response.json()["data"], key=lambda d: d["index"])
        if len(data) != len(batch):
            raise RuntimeError(
                f"OpenRouter returned {len(data)} embeddings "
                f"for a batch of {len(batch)} texts."
            )
        return [np.asarray(d["embedding"], dtype=np.float32) for d in data]
