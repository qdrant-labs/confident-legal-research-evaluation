import logging
from enum import StrEnum

from pydantic import BaseModel
from ranx import Qrels, Run, evaluate
from tqdm.auto import tqdm

from src.dataset import ClercSplit
from src.search import Searcher

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
