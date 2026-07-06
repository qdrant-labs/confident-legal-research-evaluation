import logging
from enum import StrEnum

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict
from ranx import Qrels, Run, compare, evaluate
from ranx.data_structures.report import Report
from scipy.stats import wilcoxon
from tqdm.auto import tqdm

from src.dataset import ClercSplit
from src.search import Searcher

PairwiseTest = Literal["wilcoxon", "permutation"]

_METRIC_DISPLAY = {
    "ndcg": "NDCG",
    "mrr": "MRR",
    "map": "MAP",
    "recall": "Recall",
    "precision": "Precision",
    "hits": "Hits",
}


def _display_metric(m: "RetrievalMetric") -> str:
    name, cutoff = m.value.split("@")
    return f"{_METRIC_DISPLAY.get(name, name.upper())}@{cutoff}"

logger = logging.getLogger(__name__)


class RetrievalMetric(StrEnum):
    NDCG_10 = "ndcg@10"
    RECALL_10 = "recall@10"
    MRR_10 = "mrr@10"
    NDCG_100 = "ndcg@100"
    MRR_100 = "mrr@100"
    RECALL_100 = "recall@100"


DEFAULT_METRICS = [
    RetrievalMetric.NDCG_10,
    RetrievalMetric.RECALL_10,
    RetrievalMetric.MRR_10,
]


class RetrievalReport(BaseModel):
    """Retrieval metrics over a query set, plus enough context to compare runs."""

    metrics: dict[RetrievalMetric, float]
    num_queries: int
    top_k: int

    def __str__(self) -> str:
        lines = [f"Retrieval evaluation ({self.num_queries} queries, top-{self.top_k}):"]
        for name, value in self.metrics.items():
            lines.append(f"  {name:<12} {value:.4f}")
        return "\n".join(lines)

def _max_cutoff(metrics: list[RetrievalMetric]) -> int:
    return max(int(m.value.split("@")[1]) for m in metrics)

def evaluate_retrieval(
    searcher: Searcher,
    data: ClercSplit,
    *,
    metrics: list[RetrievalMetric] = DEFAULT_METRICS,
    top_k: int = 10,
    limit: int | None = None,
) -> RetrievalReport:
    """Run every query through `searcher` and score against CLERC qrels.

    `limit` caps the number of queries (ordered by qrels iteration) for quick
    smoke runs; None evaluates the full split.
    """
    top_k = max(top_k, _max_cutoff(metrics))
    logger.info(f"It is top {top_k} evaluation")
    query_ids = list(data.qrels)
    if limit is not None:
        query_ids = query_ids[:limit]

    run_dict: dict[str, dict[str, float]] = {}
    for qid in tqdm(query_ids, desc="search"):
        hits = searcher.search(data.queries[qid].query, limit=top_k)
        run_dict[qid] = {hit.doc_id: hit.score for hit in hits}

    qrels = Qrels({qid: {did: 1 for did in data.qrels[qid]} for qid in query_ids})
    scores = evaluate(qrels, Run(run_dict), [m.value for m in metrics])
    if isinstance(scores, float):
        scores = {metrics[0].value: scores}

    return RetrievalReport(
        metrics={m: float(scores[m.value]) for m in metrics},
        num_queries=len(query_ids),
        top_k=top_k,
    )


class PairwiseMetric(BaseModel):
    """Pairwise stats for a single metric between run B and run A."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metric: RetrievalMetric
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    win_rate: float


class PairwiseComparisonReport(BaseModel):
    """Formatted 2-run comparison: mean deltas, bootstrap CIs, permutation p-values, win rates."""

    name_a: str
    name_b: str
    num_queries: int
    top_k: int
    ci_level: float
    n_bootstrap: int
    n_permutations: int
    metrics: list[PairwiseMetric]

    def __str__(self) -> str:
        label_width = max(len(_display_metric(m.metric)) for m in self.metrics) + 1
        lines = [
            f"{self.name_b} vs {self.name_a} "
            f"({self.num_queries} queries, top-{self.top_k}, "
            f"{int(self.ci_level * 100)}% CI):"
        ]
        for m in self.metrics:
            label = f"{_display_metric(m.metric)}:"
            lines.append(
                f"  {label:<{label_width + 1}} "
                f"Δ = {m.delta:+.3f}   "
                f"{int(self.ci_level * 100)}% CI [{m.ci_low:+.3f}, {m.ci_high:+.3f}]   "
                f"p = {m.p_value:.1e}   "
                f"{self.name_b}>{self.name_a} in {m.win_rate * 100:.1f}% of queries"
            )
        return "\n".join(lines)


def _pairwise_stats(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    n_bootstrap: int,
    n_permutations: int,
    ci_level: float,
    stat_test: PairwiseTest,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, float]:
    """Compute (delta, ci_low, ci_high, p_value, win_rate) for paired scores.

    `stat_test="wilcoxon"` uses scipy's signed-rank test — asymptotic normal
    approximation, resolves p-values well below 1e-10 at large N. `"permutation"`
    uses paired sign-flip randomization; floor at ~1/n_permutations.
    """
    diffs = scores_b - scores_a
    n = len(diffs)
    delta = float(diffs.mean())

    boot_means = rng.choice(diffs, size=(n_bootstrap, n), replace=True).mean(axis=1)
    alpha = 1.0 - ci_level
    ci_low = float(np.quantile(boot_means, alpha / 2))
    ci_high = float(np.quantile(boot_means, 1 - alpha / 2))

    if stat_test == "wilcoxon":
        if np.all(diffs == 0):
            p_value = 1.0
        else:
            result = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
            p_value = float(result.pvalue)
    else:
        signs = rng.choice([-1, 1], size=(n_permutations, n))
        perm_means = (signs * diffs).mean(axis=1)
        hits = int(np.sum(np.abs(perm_means) >= abs(delta)))
        p_value = (hits + 1) / (n_permutations + 1)

    win_rate = float((scores_b > scores_a).mean())
    return delta, ci_low, ci_high, p_value, win_rate


def compare_pair(
    searchers: dict[str, Searcher],
    data: ClercSplit,
    *,
    metrics: list[RetrievalMetric] = DEFAULT_METRICS,
    top_k: int = 10,
    limit: int | None = None,
    stat_test: PairwiseTest = "wilcoxon",
    n_bootstrap: int = 10000,
    n_permutations: int = 10000,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> PairwiseComparisonReport:
    """Detailed pairwise comparison: mean delta, bootstrap CI, permutation p, win rate.

    Insertion order of `searchers` matters: first key is baseline A, second is B.
    Deltas are `mean(B) - mean(A)`; positive means B is better.
    """
    if len(searchers) != 2:
        raise ValueError(
            f"compare_pair requires exactly two searchers, got {len(searchers)}."
        )
    top_k = max(top_k, _max_cutoff(metrics))
    name_a, name_b = list(searchers)
    query_ids = list(data.qrels)
    if limit is not None:
        query_ids = query_ids[:limit]

    qrels = Qrels({qid: {did: 1 for did in data.qrels[qid]} for qid in query_ids})
    metric_names = [m.value for m in metrics]
    per_query: dict[str, dict[str, np.ndarray]] = {}
    for name, searcher in searchers.items():
        run_dict: dict[str, dict[str, float]] = {}
        for qid in tqdm(query_ids, desc=f"search:{name}"):
            hits = searcher.search(data.queries[qid].query, limit=top_k)
            run_dict[qid] = {hit.doc_id: hit.score for hit in hits}
        run = Run(run_dict)
        run.name = name
        per_query[name] = evaluate(qrels, run, metric_names, return_mean=False)

    rng = np.random.default_rng(random_seed)
    results: list[PairwiseMetric] = []
    for m in metrics:
        scores_a = per_query[name_a][m.value]
        scores_b = per_query[name_b][m.value]
        delta, ci_low, ci_high, p, win_rate = _pairwise_stats(
            scores_a,
            scores_b,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            ci_level=ci_level,
            stat_test=stat_test,
            rng=rng,
        )
        results.append(
            PairwiseMetric(
                metric=m,
                mean_a=float(scores_a.mean()),
                mean_b=float(scores_b.mean()),
                delta=delta,
                ci_low=ci_low,
                ci_high=ci_high,
                p_value=p,
                win_rate=win_rate,
            )
        )

    return PairwiseComparisonReport(
        name_a=name_a,
        name_b=name_b,
        num_queries=len(query_ids),
        top_k=top_k,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        metrics=results,
    )


def compare_retrievers(
    searchers: dict[str, Searcher],
    data: ClercSplit,
    *,
    metrics: list[RetrievalMetric] = DEFAULT_METRICS,
    top_k: int = 10,
    stat_test: str = "fisher",
    max_p: float = 0.05,
    limit: int | None = None,
) -> Report:
    """Run every searcher over the same queries and pairwise-compare with ranx.

    Requires paired evaluation: identical `qid` set and `top_k` across runs so
    the significance test is valid. `stat_test="fisher"` is a paired randomization
    test — non-parametric, safe for bounded IR metrics. With 2 runs no multiple-
    comparison correction is needed; with 3+ runs consider `stat_test="tukey"`.
    """
    top_k = max(top_k, _max_cutoff(metrics))
    logger.info(f"Comparing {len(searchers)} retrievers at top-{top_k}")
    query_ids = list(data.qrels)
    if limit is not None:
        query_ids = query_ids[:limit]

    runs: list[Run] = []
    for name, searcher in searchers.items():
        run_dict: dict[str, dict[str, float]] = {}
        for qid in tqdm(query_ids, desc=f"search:{name}"):
            hits = searcher.search(data.queries[qid].query, limit=top_k)
            run_dict[qid] = {hit.doc_id: hit.score for hit in hits}
        run = Run(run_dict)
        run.name = name
        runs.append(run)

    qrels = Qrels({qid: {did: 1 for did in data.qrels[qid]} for qid in query_ids})
    return compare(
        qrels=qrels,
        runs=runs,
        metrics=[m.value for m in metrics],
        stat_test=stat_test,
        max_p=max_p,
    )
