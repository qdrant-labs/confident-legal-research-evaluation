import logging
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.dataset import Document

logger = logging.getLogger(__name__)

_THRESHOLDS = (128, 256, 512, 1024, 2048)


class TokenLengthQuantiles(BaseModel):
    p50: int
    p90: int
    p95: int
    p99: int
    max: int


class CorpusStats(BaseModel):
    """Corpus characteristics for deciding whether chunking is needed.

    `max_seq_length` is the context window of the embedding model being
    considered. `pct_over_max` and `truncation_loss_pct` answer the primary
    question: does the corpus overflow this model, and if so how badly?
    """

    tokenizer: str
    max_seq_length: int
    num_documents: int
    total_tokens: int
    mean_tokens: float
    quantiles: TokenLengthQuantiles
    pct_over_max: float
    pct_over_threshold: dict[int, float]
    truncation_loss_pct: float

    def __str__(self) -> str:
        q = self.quantiles
        lines = [
            f"Corpus stats ({self.num_documents:,} docs, tokenizer={self.tokenizer}):",
            f"  Total tokens:      {self.total_tokens:,}",
            f"  Mean tokens/doc:   {self.mean_tokens:.1f}",
            f"  Length quantiles:  p50={q.p50}  p90={q.p90}  p95={q.p95}  p99={q.p99}  max={q.max}",
            f"  Over model max ({self.max_seq_length} tok): "
            f"{self.pct_over_max * 100:.1f}% of docs",
            f"  Truncation info loss: "
            f"{self.truncation_loss_pct * 100:.1f}% of total tokens discarded",
            "  Length distribution:",
        ]
        for threshold in sorted(self.pct_over_threshold):
            pct = self.pct_over_threshold[threshold]
            lines.append(f"    > {threshold:>4d} tok: {pct * 100:5.1f}%")
        return "\n".join(lines)

    def needs_chunking(self, over_max_pct: float = 0.10, loss_pct: float = 0.10) -> bool:
        """Simple decision rule: chunk if >10% of docs overflow the window
        or >10% of total tokens are being silently discarded."""
        return (
            self.pct_over_max > over_max_pct
            or self.truncation_loss_pct > loss_pct
        )


def analyze_corpus(
    documents: Sequence[Document],
    tokenizer_name: str = "BAAI/bge-base-en-v1.5",
    max_seq_length: int = 512,
    batch_size: int = 1000,
) -> CorpusStats:
    """Tokenize the corpus and report length distribution vs `max_seq_length`.

    Uses the actual model tokenizer (not char/word counts) so results reflect
    what the embedding model will see. Batch-encodes for speed; ~1-2 min on
    a 40k-doc corpus with a fast HF tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        model_max_length=int(1e30),
    )
    lengths: list[int] = []
    for start in tqdm(range(0, len(documents), batch_size), desc="tokenize"):
        batch = documents[start : start + batch_size]
        encoded = tokenizer(
            [d.text for d in batch],
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])

    arr = np.asarray(lengths, dtype=np.int64)
    total_tokens = int(arr.sum())
    truncated = int(np.maximum(arr - max_seq_length, 0).sum())
    truncation_loss_pct = truncated / total_tokens if total_tokens else 0.0

    return CorpusStats(
        tokenizer=tokenizer_name,
        max_seq_length=max_seq_length,
        num_documents=len(documents),
        total_tokens=total_tokens,
        mean_tokens=float(arr.mean()),
        quantiles=TokenLengthQuantiles(
            p50=int(np.percentile(arr, 50)),
            p90=int(np.percentile(arr, 90)),
            p95=int(np.percentile(arr, 95)),
            p99=int(np.percentile(arr, 99)),
            max=int(arr.max()),
        ),
        pct_over_max=float((arr > max_seq_length).mean()),
        pct_over_threshold={t: float((arr > t).mean()) for t in _THRESHOLDS},
        truncation_loss_pct=truncation_loss_pct,
    )
