import logging
import math
import re
import zlib
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
from pydantic import BaseModel
from qdrant_client.models import SparseVector
from tqdm.auto import tqdm

from src.dataset import ClercSplit, Document

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9§][a-z0-9§.\-()]*")
_QUOTE_CHARS = ('"', "“", "”")
_GRAM_SEP = "_"


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens that preserve legal identifiers (f.2d, § 512(k))."""
    return [t.rstrip(".") for t in _TOKEN.findall(text.lower())]


def ngrams(tokens: list[str], max_n: int) -> list[str]:
    """All word n-grams of order 1..max_n; higher orders joined with '_'.

    The token pattern cannot produce '_', so a gram's order is recoverable as
    `gram.count('_') + 1` — which is how lower-order arms reuse the combined
    feature space without a second tokenization pass.
    """
    grams = list(tokens)
    for n in range(2, max_n + 1):
        grams.extend(
            _GRAM_SEP.join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
        )
    return grams


class NgramStats:
    """Precomputed corpus BM25 statistics for n-gram scoring.

    IDF is document-frequency based over the full corpus, restricted to grams
    occurring in the provided queries so memory stays proportional to the
    query set, not the corpus vocabulary. Grams unseen at build time score
    with the maximum IDF (df = 0). Shared by the separation diagnostic
    (`analyze_ngram_signal`) and the `NgramBM25Strategy` reranker.
    """

    def __init__(
        self,
        idf: dict[str, float],
        mean_tokens: float,
        corpus_size: int,
        max_n: int,
        k1: float,
        b: float,
    ) -> None:
        self.idf = idf
        self.mean_tokens = mean_tokens
        self.corpus_size = corpus_size
        self.max_n = max_n
        self.k1 = k1
        self.b = b
        self._default_idf = math.log(1 + (corpus_size + 0.5) / 0.5)

    @classmethod
    def build(
        cls,
        corpus: dict[str, Document],
        queries: Sequence[str],
        max_n: int = 2,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "NgramStats":
        """One corpus pass: document frequencies for query grams + average length."""
        wanted: set[str] = set()
        for query in queries:
            wanted.update(ngrams(tokenize(query), max_n))

        doc_freq: Counter = Counter()
        token_lengths: list[int] = []
        for doc in tqdm(corpus.values(), desc="corpus df"):
            tokens = tokenize(doc.text)
            token_lengths.append(len(tokens))
            doc_freq.update(set(ngrams(tokens, max_n)) & wanted)

        corpus_size = len(corpus)
        idf = {
            g: math.log(1 + (corpus_size - doc_freq[g] + 0.5) / (doc_freq[g] + 0.5))
            for g in wanted
        }
        return cls(
            idf=idf,
            mean_tokens=float(np.mean(token_lengths)),
            corpus_size=corpus_size,
            max_n=max_n,
            k1=k1,
            b=b,
        )

    def text_grams(self, text: str, n: int | None = None) -> set[str]:
        """Unique grams of a query text at orders 1..n (default: build-time max_n)."""
        return set(ngrams(tokenize(text), n or self.max_n))

    def avgdl(self, n: int) -> float:
        """Average document length in grams for orders 1..n."""
        return sum(self.mean_tokens - j + 1 for j in range(1, n + 1))

    def score(
        self,
        query_grams: set[str],
        doc_counts: Counter,
        doc_num_tokens: int,
        n: int | None = None,
    ) -> float:
        """BM25 score of one document against a query gram set at orders 1..n.

        `doc_counts` may contain higher-order grams than `n`; the query gram
        set is what restricts which orders contribute.
        """
        n = n or self.max_n
        doc_len = sum(doc_num_tokens - j + 1 for j in range(1, n + 1))
        norm = self.k1 * (1 - self.b + self.b * doc_len / self.avgdl(n))
        return sum(
            self.idf.get(g, self._default_idf) * tf * (self.k1 + 1) / (tf + norm)
            for g in query_grams
            if (tf := doc_counts.get(g))
        )


class NgramSparseEncoder:
    """Client-side sparse encoder: hashed word n-grams with BM25-saturated TF.

    Document values carry the saturated term frequency (k1/b applied at
    encode time using the corpus average length); query values are 1.0 —
    pair the slot with `Modifier.IDF` so Qdrant applies collection-wide IDF
    at query time, replacing any local IDF table. Gram indices are crc32
    hashes (uint32); collisions (~0.2% at CLERC scale) accumulate values.
    """

    def __init__(
        self,
        max_n: int = 2,
        k1: float = 1.5,
        b: float = 0.75,
        avg_tokens: float | None = None,
    ) -> None:
        self.max_n = max_n
        self.k1 = k1
        self.b = b
        self.avg_tokens = avg_tokens

    @classmethod
    def fit(
        cls,
        texts: Iterable[str],
        max_n: int = 2,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "NgramSparseEncoder":
        """Compute the corpus average token length needed for document encoding."""
        lengths = [len(tokenize(t)) for t in tqdm(texts, desc="ngram fit")]
        return cls(max_n=max_n, k1=k1, b=b, avg_tokens=float(np.mean(lengths)))

    @staticmethod
    def _hash(gram: str) -> int:
        return zlib.crc32(gram.encode())

    def _gram_len(self, num_tokens: int) -> float:
        return sum(num_tokens - j + 1 for j in range(1, self.max_n + 1))

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        if self.avg_tokens is None:
            raise ValueError(
                "Document encoding needs `avg_tokens`: build the encoder via "
                "NgramSparseEncoder.fit(corpus_texts) or pass avg_tokens."
            )
        avgdl = self._gram_len(round(self.avg_tokens))
        vectors: list[SparseVector] = []
        for text in tqdm(texts, desc="embed:ngram"):
            tokens = tokenize(text)
            counts = Counter(ngrams(tokens, self.max_n))
            norm = self.k1 * (
                1 - self.b + self.b * self._gram_len(len(tokens)) / avgdl
            )
            features: dict[int, float] = {}
            for gram, tf in counts.items():
                idx = self._hash(gram)
                saturated = tf * (self.k1 + 1) / (tf + norm)
                features[idx] = features.get(idx, 0.0) + saturated
            vectors.append(
                SparseVector(indices=list(features), values=list(features.values()))
            )
        return vectors

    def encode_query(self, text: str) -> SparseVector:
        indices = {self._hash(g) for g in set(ngrams(tokenize(text), self.max_n))}
        return SparseVector(indices=list(indices), values=[1.0] * len(indices))


class NgramArmStats(BaseModel):
    """Gold-vs-hard-negatives separation for one scoring arm on one query stratum."""

    arm: str
    stratum: str
    num_queries: int
    mean_relative_rank: float
    gold_top_k_pct: dict[int, float]
    gold_zero_overlap_pct: float


class NgramSignalReport(BaseModel):
    """Does contiguity (bigrams) add discrimination beyond vocabulary (unigrams)?

    Both arms score gold against each query's own mined hard negatives with
    corpus-IDF BM25 over the same tokenization; the only difference is the
    maximum gram order. The hypothesis lives in the *delta between arms*,
    not in either arm's absolute numbers. Strata split queries by a cheap
    direct-quote proxy (quotation marks present), predicting that bigram
    gains concentrate where the citing court quotes the cited case.
    """

    k1: float
    b: float
    num_queries: int
    median_pool_size: int
    random_top1_pct: float
    arms: list[NgramArmStats]

    def __str__(self) -> str:
        lines = [
            f"N-gram signal ({self.num_queries:,} queries, "
            f"median pool {self.median_pool_size}, BM25 k1={self.k1} b={self.b}, "
            f"gold ranked among own hard negatives):",
            f"  {'arm':<10}{'stratum':<11}{'queries':>8}{'rel.rank':>10}"
            f"{'top-1':>8}{'top-3':>8}{'top-5':>8}{'zero-ovl':>10}",
        ]
        for a in self.arms:
            lines.append(
                f"  {a.arm:<10}{a.stratum:<11}{a.num_queries:>8,}"
                f"{a.mean_relative_rank:>10.3f}"
                f"{a.gold_top_k_pct.get(1, 0):>8.1%}"
                f"{a.gold_top_k_pct.get(3, 0):>8.1%}"
                f"{a.gold_top_k_pct.get(5, 0):>8.1%}"
                f"{a.gold_zero_overlap_pct:>10.1%}"
            )
        lines.append(
            f"  (random top-1 baseline: {self.random_top1_pct:.1%}; "
            f"random rel.rank: 0.500)"
        )
        return "\n".join(lines)


def analyze_ngram_signal(
    data: ClercSplit,
    max_n: int = 2,
    k1: float = 1.5,
    b: float = 0.75,
    top_ks: tuple[int, ...] = (1, 3, 5),
) -> NgramSignalReport:
    """Rank gold among each query's hard negatives with BM25 at gram orders 1..max_n.

    One arm per gram order: arm n scores with all grams of order <= n, so
    arm 1 is classic unigram BM25 (the control) and arm max_n carries the
    contiguity hypothesis. IDF and average length come from a shared
    `NgramStats` built over the full pooled corpus.
    """
    stats = NgramStats.build(
        data.corpus,
        [q.query for q in data.queries.values()],
        max_n=max_n,
        k1=k1,
        b=b,
    )
    query_grams = {
        qid: stats.text_grams(q.query) for qid, q in data.queries.items()
    }
    arm_grams = {
        n: {
            qid: {g for g in gs if g.count(_GRAM_SEP) < n}
            for qid, gs in query_grams.items()
        }
        for n in range(1, max_n + 1)
    }

    # per-query records: (quoted, relative_rank, absolute_rank, zero_overlap)
    ranks: dict[int, list[tuple[bool, float, float, bool]]] = {
        n: [] for n in range(1, max_n + 1)
    }
    pool_sizes: list[int] = []
    for qid, gold_pids in tqdm(data.qrels.items(), desc="ngram-rank"):
        negative_pids = data.negatives.get(qid, set())
        if not negative_pids:
            continue
        candidates = [*gold_pids, *negative_pids]
        pool_sizes.append(len(candidates))
        quoted = any(ch in data.queries[qid].query for ch in _QUOTE_CHARS)

        cand_features = {}
        for pid in candidates:
            tokens = tokenize(data.corpus[pid].text)
            cand_features[pid] = (Counter(ngrams(tokens, max_n)), len(tokens))

        for n in range(1, max_n + 1):
            grams_q = arm_grams[n][qid]
            scores = {
                pid: stats.score(grams_q, counts, num_tokens, n=n)
                for pid, (counts, num_tokens) in cand_features.items()
            }
            gold_best = max(scores[pid] for pid in gold_pids)
            better = sum(1 for s in scores.values() if s > gold_best)
            tied = sum(1 for s in scores.values() if s == gold_best) - 1
            rank = better + 1 + tied / 2
            ranks[n].append(
                (quoted, rank / len(candidates), rank, gold_best == 0)
            )

    pools = np.asarray(pool_sizes)
    arm_labels = {n: "unigram" if n == 1 else f"uni+{n}gram" for n in ranks}
    arms: list[NgramArmStats] = []
    for n, records in ranks.items():
        for stratum, keep in (
            ("all", lambda q: True),
            ("quoted", lambda q: q),
            ("unquoted", lambda q: not q),
        ):
            subset = [r for r in records if keep(r[0])]
            if not subset:
                continue
            rel = np.asarray([r[1] for r in subset])
            abs_rank = np.asarray([r[2] for r in subset])
            arms.append(
                NgramArmStats(
                    arm=arm_labels[n],
                    stratum=stratum,
                    num_queries=len(subset),
                    mean_relative_rank=float(rel.mean()),
                    gold_top_k_pct={
                        k: float((abs_rank <= k).mean()) for k in top_ks
                    },
                    gold_zero_overlap_pct=float(
                        np.mean([r[3] for r in subset])
                    ),
                )
            )

    return NgramSignalReport(
        k1=k1,
        b=b,
        num_queries=len(pool_sizes),
        median_pool_size=int(np.median(pools)),
        random_top1_pct=float((1 / pools).mean()),
        arms=arms,
    )
