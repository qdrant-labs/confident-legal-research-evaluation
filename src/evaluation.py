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
from src.doc_mapping import build_did_to_pids
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
        lines = [
            f"Retrieval evaluation ({self.num_queries} queries, top-{self.top_k}):"
        ]
        for name, value in self.metrics.items():
            lines.append(f"  {name:<12} {value:.4f}")
        return "\n".join(lines)

def _max_cutoff(metrics: list[RetrievalMetric]) -> int:
    return max(int(m.value.split("@")[1]) for m in metrics)


def _build_run(
    searcher: Searcher,
    data: ClercSplit,
    query_ids: list[str],
    top_k: int,
    desc: str = "search",
    qrels: dict[str, dict[str, int]] | None = None,
    metrics: list[RetrievalMetric] | None = None,
    progress_every: int = 50,
) -> dict[str, dict[str, float]]:
    """Run every query through `searcher` and collect doc_id->score maps.

    When `qrels` and `metrics` are given, running metric values over the
    queries processed so far are shown on the progress bar (refreshed every
    `progress_every` queries), so long evaluations are observable mid-flight.
    """
    run_dict: dict[str, dict[str, float]] = {}
    progress = tqdm(query_ids, desc=desc)
    for i, qid in enumerate(progress, start=1):
        hits = searcher.search(data.queries[qid].query, limit=top_k)
        run_dict[qid] = {hit.doc_id: hit.score for hit in hits}
        if (
            qrels is not None
            and metrics is not None
            and (i % progress_every == 0 or i == len(query_ids))
        ):
            partial_qrels = Qrels({q: qrels[q] for q in run_dict})
            scores = evaluate(
                partial_qrels, Run(dict(run_dict)), [m.value for m in metrics]
            )
            if isinstance(scores, float):
                scores = {metrics[0].value: scores}
            progress.set_postfix({k: f"{v:.3f}" for k, v in scores.items()})
    return run_dict

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

    qrels_dict = {qid: {did: 1 for did in data.qrels[qid]} for qid in query_ids}
    run_dict = _build_run(
        searcher, data, query_ids, top_k, qrels=qrels_dict, metrics=metrics
    )

    scores = evaluate(Qrels(qrels_dict), Run(run_dict), [m.value for m in metrics])
    if isinstance(scores, float):
        scores = {metrics[0].value: scores}

    return RetrievalReport(
        metrics={m: float(scores[m.value]) for m in metrics},
        num_queries=len(query_ids),
        top_k=top_k,
    )


def build_sibling_qrels(
    data: ClercSplit,
    pid2did: dict[str, str],
    positive_grade: int = 2,
    sibling_grade: int = 1,
) -> dict[str, dict[str, int]]:
    """Graded qrels that give partial credit for 'right document, wrong passage'.

    Labeled positive passages get `positive_grade`; other pool passages of the
    same source document (siblings) get `sibling_grade`. Under strict
    passage-level qrels those siblings are document-level false negatives and
    score zero.
    """
    did_to_pids = build_did_to_pids(data.corpus, pid2did)

    qrels: dict[str, dict[str, int]] = {}
    for qid, pos_pids in data.qrels.items():
        graded: dict[str, int] = {}
        for pid in pos_pids:
            for sibling in did_to_pids[pid2did[pid]]:
                graded[sibling] = sibling_grade
        for pid in pos_pids:
            graded[pid] = positive_grade
        qrels[qid] = graded
    return qrels


def evaluate_retrieval_graded(
    searcher: Searcher,
    data: ClercSplit,
    pid2did: dict[str, str],
    *,
    metrics: list[RetrievalMetric] = DEFAULT_METRICS,
    top_k: int = 10,
    limit: int | None = None,
    positive_grade: int = 2,
    sibling_grade: int = 1,
) -> RetrievalReport:
    """Sibling-aware variant of `evaluate_retrieval`; same report, graded qrels.

    NDCG grants partial credit for retrieving a sibling of the positive;
    Recall/MRR treat any graded passage (positive or sibling) as relevant, so
    they become document-level lenient. Run next to `evaluate_retrieval` on
    the same searcher to size the passage-labeling artifact per metric.
    """
    top_k = max(top_k, _max_cutoff(metrics))
    logger.info(f"It is top {top_k} graded evaluation")
    query_ids = list(data.qrels)
    if limit is not None:
        query_ids = query_ids[:limit]

    graded = build_sibling_qrels(data, pid2did, positive_grade, sibling_grade)
    qrels_dict = {qid: graded[qid] for qid in query_ids}
    run_dict = _build_run(
        searcher, data, query_ids, top_k, qrels=qrels_dict, metrics=metrics
    )

    scores = evaluate(Qrels(qrels_dict), Run(run_dict), [m.value for m in metrics])
    if isinstance(scores, float):
        scores = {metrics[0].value: scores}

    return RetrievalReport(
        metrics={m: float(scores[m.value]) for m in metrics},
        num_queries=len(query_ids),
        top_k=top_k,
    )


class DocLevelReport(BaseModel):
    """Document-level diagnostics of top-k retrieval — the reranker headroom check.

    Each query is classified by its best outcome in top-k: 'exact' (a labeled
    positive passage was retrieved), 'sibling_only' (only other passages of
    the positive document made it — right document, wrong slice), 'miss' (the
    document is absent entirely). `mean_doc_precision` is the average fraction
    of returned hits belonging to the positive document. A high sibling_only
    share is headroom a sibling-aware reranker can convert into exact hits;
    a high miss share means the retriever, not the labeling, is the problem.
    """

    num_queries: int
    top_k: int
    exact_query_ids: list[str]
    sibling_only_query_ids: list[str]
    miss_query_ids: list[str]
    mean_doc_precision: float

    def __str__(self) -> str:
        exact = len(self.exact_query_ids)
        sibling = len(self.sibling_only_query_ids)
        miss = len(self.miss_query_ids)
        doc_hits = exact + sibling
        n = self.num_queries
        return "\n".join(
            [
                f"Document-level diagnosis ({n} queries, top-{self.top_k}):",
                f"  doc hit rate (any slice of positive doc): "
                f"{doc_hits} ({doc_hits / n:.1%})",
                f"    exact positive retrieved:               "
                f"{exact} ({exact / n:.1%})",
                f"    sibling only (right doc, wrong slice):  "
                f"{sibling} ({sibling / n:.1%})",
                f"  full miss (document absent):              "
                f"{miss} ({miss / n:.1%})",
                f"  mean doc precision@{self.top_k}:          "
                f"{self.mean_doc_precision:.3f}",
            ]
        )


def diagnose_doc_level(
    searcher: Searcher,
    data: ClercSplit,
    pid2did: dict[str, str],
    *,
    top_k: int = 10,
    limit: int | None = None,
    progress_every: int = 50,
) -> DocLevelReport:
    """One search pass; classify each query's failure mode at document level.

    Unlike NDCG — even graded — this counts events instead of averaging
    rank-discounted scores, so it answers directly: how often is the right
    document retrieved via the wrong passage? Query ids are kept per bucket
    for eyeballing examples. Running bucket rates are shown on the progress
    bar every `progress_every` queries.
    """
    query_ids = list(data.qrels)
    if limit is not None:
        query_ids = query_ids[:limit]

    exact_ids: list[str] = []
    sibling_only_ids: list[str] = []
    miss_ids: list[str] = []
    precisions: list[float] = []
    progress = tqdm(query_ids, desc="diagnose")
    for i, qid in enumerate(progress, start=1):
        hits = searcher.search(data.queries[qid].query, limit=top_k)
        pos_pids = data.qrels[qid]
        pos_dids = {pid2did[pid] for pid in pos_pids}
        hit_pids = [hit.doc_id for hit in hits]
        in_doc = [pid for pid in hit_pids if pid2did[pid] in pos_dids]
        precisions.append(len(in_doc) / len(hit_pids) if hit_pids else 0.0)
        if any(pid in pos_pids for pid in hit_pids):
            exact_ids.append(qid)
        elif in_doc:
            sibling_only_ids.append(qid)
        else:
            miss_ids.append(qid)
        if i % progress_every == 0 or i == len(query_ids):
            progress.set_postfix(
                {
                    "exact": f"{len(exact_ids) / i:.1%}",
                    "sibling": f"{len(sibling_only_ids) / i:.1%}",
                    "miss": f"{len(miss_ids) / i:.1%}",
                    "doc_p": f"{np.mean(precisions):.3f}",
                }
            )

    return DocLevelReport(
        num_queries=len(query_ids),
        top_k=top_k,
        exact_query_ids=exact_ids,
        sibling_only_query_ids=sibling_only_ids,
        miss_query_ids=miss_ids,
        mean_doc_precision=float(np.mean(precisions)) if precisions else 0.0,
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
    loss_rate: float

    @property
    def tie_rate(self) -> float:
        return 1.0 - self.win_rate - self.loss_rate


class PairwiseComparisonReport(BaseModel):
    """2-run comparison: mean deltas, bootstrap CIs, p-values, win/loss/tie rates."""

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
                f"{self.name_b}>{self.name_a}: {m.win_rate:.1%} | "
                f"{self.name_a}>{self.name_b}: {m.loss_rate:.1%} | "
                f"ties: {m.tie_rate:.1%}"
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
) -> tuple[float, float, float, float]:
    """Compute (delta, ci_low, ci_high, p_value) for paired scores.

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

    return delta, ci_low, ci_high, p_value


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
        run_dict = _build_run(searcher, data, query_ids, top_k, desc=f"search:{name}")
        run = Run(run_dict)
        run.name = name
        per_query[name] = evaluate(qrels, run, metric_names, return_mean=False)

    rng = np.random.default_rng(random_seed)
    results: list[PairwiseMetric] = []
    for m in metrics:
        scores_a = per_query[name_a][m.value]
        scores_b = per_query[name_b][m.value]
        delta, ci_low, ci_high, p = _pairwise_stats(
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
                win_rate=float((scores_b > scores_a).mean()),
                loss_rate=float((scores_a > scores_b).mean()),
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
        run_dict = _build_run(searcher, data, query_ids, top_k, desc=f"search:{name}")
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
