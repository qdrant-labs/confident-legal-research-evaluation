import logging
import re
from collections import Counter

import numpy as np
from pydantic import BaseModel
from tqdm.auto import tqdm

from src.dataset import ClercSplit

logger = logging.getLogger(__name__)

_REPORTER = (
    r"\d+\s+(?:U\.S\.|S\.\s?Ct\.|F\.(?:2d|3d)?|F\.\s?Supp\.(?:\s?2d)?"
    r"|Stat\.|L\.\s?Ed\.(?:\s?2d)?|B\.R\.|A\.(?:2d)?|P\.(?:2d|3d)?"
    r"|N\.E\.(?:2d)?|N\.W\.(?:2d)?|S\.E\.(?:2d)?|S\.W\.(?:2d|3d)?|So\.(?:2d)?)\s+\d+"
)
_PATTERNS: dict[str, re.Pattern[str]] = {
    "usc": re.compile(r"\d+\s+U\.S\.C\.\s*§+\s*[\dA-Za-z().\-]+"),
    "reporter": re.compile(_REPORTER),
    "section": re.compile(r"§+\s*\d[\dA-Za-z().\-]*"),
    # curly apostrophe is deliberate: CLERC text uses typographic quotes
    "case": re.compile(r"[A-Z][A-Za-z'’.\-]+\s+v\.\s+[A-Z][A-Za-z'’.\-]+"),  # noqa: RUF001
}
_WHITESPACE = re.compile(r"\s+")


def extract_entities(text: str) -> set[str]:
    """Extract normalized legal identifiers from text.

    Covers statute cites (17 U.S.C. § 512(k)), reporter cites (112 Stat. 2877),
    bare section refs (§ 512(d)) and case names (Doe v. Roe), each tagged by
    kind so identical numbers of different kinds don't collide. Regex is the
    cheap first pass; eyecite is the upgrade path if the signal proves out.
    """
    return {
        f"{kind}:{_WHITESPACE.sub(' ', match.strip().lower())}"
        for kind, pattern in _PATTERNS.items()
        for match in pattern.findall(text)
    }


class EntitySignalStats(BaseModel):
    """Does entity overlap separate gold from hard negatives?

    For each query, the gold passage is ranked among the query's own mined
    hard negatives by IDF-weighted overlap with the query's entities. Hard
    negatives stand in for a retrieved pool: they are exactly the distractors
    that fooled first-stage retrieval. `mean_relative_rank` is 0 when gold
    always wins, 0.5 for a random signal; `gold_top_k_pct` is the headroom an
    entity-based reranker could exploit.
    """

    num_queries: int
    queries_with_entities: int
    median_entities_per_query: int
    median_pool_size: int
    gold_zero_overlap_pct: float
    mean_relative_rank: float
    gold_top_k_pct: dict[int, float]
    random_top1_pct: float

    def __str__(self) -> str:
        entity_pct = self.queries_with_entities / self.num_queries
        lines = [
            f"Entity signal ({self.num_queries:,} queries, "
            f"gold ranked among own hard negatives):",
            f"  Queries with >=1 entity:      "
            f"{self.queries_with_entities:,} ({entity_pct:.1%}), "
            f"median {self.median_entities_per_query} entities/query",
            f"  Median pool size (gold+negs): {self.median_pool_size}",
            f"  Gold w/ zero entity overlap:  {self.gold_zero_overlap_pct:.1%}",
            f"  Mean relative rank of gold:   {self.mean_relative_rank:.3f} "
            f"(random = 0.500)",
        ]
        for k, pct in sorted(self.gold_top_k_pct.items()):
            lines.append(f"  Gold in top-{k} by entities:    {pct:.1%}")
        lines.append(f"  Random top-1 baseline:        {self.random_top1_pct:.1%}")
        return "\n".join(lines)


def analyze_entity_signal(
    data: ClercSplit,
    top_ks: tuple[int, ...] = (1, 3, 5),
) -> EntitySignalStats:
    """Rank each gold passage among its query's hard negatives by entity overlap.

    Overlap is IDF-weighted within the query's pool (an entity shared by every
    candidate is worthless; one shared only with gold is decisive). Ties get
    mean rank so a constant score cannot fake a good rank. Queries without
    extractable entities are counted but not ranked.
    """
    ranks: list[float] = []
    pool_sizes: list[int] = []
    entity_counts: list[int] = []
    gold_zero_overlap = 0

    for qid, gold_pids in tqdm(data.qrels.items(), desc="entity-rank"):
        negative_pids = data.negatives.get(qid, set())
        if not negative_pids:
            continue
        query_entities = extract_entities(data.queries[qid].query)
        entity_counts.append(len(query_entities))
        if not query_entities:
            continue

        candidates = [*gold_pids, *negative_pids]
        pool_sizes.append(len(candidates))

        shared = {
            pid: extract_entities(data.corpus[pid].text) & query_entities
            for pid in candidates
        }
        document_frequency = Counter(e for ents in shared.values() for e in ents)
        scores = {
            pid: sum(1.0 / document_frequency[e] for e in ents)
            for pid, ents in shared.items()
        }

        gold_best = max(scores[pid] for pid in gold_pids)
        if gold_best == 0:
            gold_zero_overlap += 1
        better = sum(1 for s in scores.values() if s > gold_best)
        tied = sum(1 for s in scores.values() if s == gold_best) - 1
        ranks.append(better + 1 + tied / 2)

    ranks_arr = np.asarray(ranks)
    pools_arr = np.asarray(pool_sizes)

    return EntitySignalStats(
        num_queries=len(entity_counts),
        queries_with_entities=len(ranks),
        median_entities_per_query=int(np.median(entity_counts)),
        median_pool_size=int(np.median(pools_arr)),
        gold_zero_overlap_pct=gold_zero_overlap / len(ranks),
        mean_relative_rank=float((ranks_arr / pools_arr).mean()),
        gold_top_k_pct={k: float((ranks_arr <= k).mean()) for k in top_ks},
        random_top1_pct=float((1 / pools_arr).mean()),
    )
