import logging
from collections import defaultdict
from collections.abc import Iterable

from huggingface_hub import hf_hub_download
from pydantic import BaseModel

from src.dataset import REPO_ID, ClercSplit

logger = logging.getLogger(__name__)

MAPPING_FILENAME = "collection/mapping.pid2did.tsv"


def load_pid2did(pids: Iterable[str]) -> dict[str, str]:
    """Map CLERC passage ids (pids) to their source document ids (dids).

    Streams the ~23.7M-line collection mapping and keeps only the requested
    pids, so memory stays proportional to the pool, not the full collection.
    """
    wanted = set(pids)
    path = hf_hub_download(
        repo_id=REPO_ID, filename=MAPPING_FILENAME, repo_type="dataset"
    )
    mapping: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            pid, did = line.rstrip("\n").split("\t")
            if pid in wanted:
                mapping[pid] = did
    if len(mapping) < len(wanted):
        logger.warning(
            f"{len(wanted) - len(mapping)} pids missing from {MAPPING_FILENAME}"
        )
    return mapping


def build_did_to_pids(
    pids: Iterable[str], pid2did: dict[str, str]
) -> dict[str, set[str]]:
    """Group pool passage ids by their source document."""
    did_to_pids: dict[str, set[str]] = defaultdict(set)
    for pid in pids:
        did_to_pids[pid2did[pid]].add(pid)
    return dict(did_to_pids)


class SiblingExample(BaseModel):
    """A query whose own hard negatives include passages of its positive document."""

    query_id: str
    positive_pids: list[str]
    sibling_negative_pids: list[str]


class SiblingStats(BaseModel):
    """Structure of the pooled corpus once passages are traced back to documents.

    'Siblings' are passages cut from the same source document. A sibling of a
    positive passage is relevant at the document level but scored irrelevant
    by passage-level qrels, so a retriever that finds the right case via the
    wrong slice gets zero credit — the labels act as false negatives.
    """

    num_passages: int
    num_documents: int
    multi_passage_documents: int
    num_queries: int
    queries_with_positive_siblings_in_pool: int
    queries_with_same_doc_negatives: int
    examples: list[SiblingExample]

    def __str__(self) -> str:
        multi_pct = self.multi_passage_documents / self.num_documents
        sibling_pct = self.queries_with_positive_siblings_in_pool / self.num_queries
        neg_pct = self.queries_with_same_doc_negatives / self.num_queries
        return "\n".join(
            [
                f"Sibling structure ({self.num_passages:,} passages, "
                f"{self.num_queries:,} queries):",
                f"  Unique source documents:               {self.num_documents:,}",
                f"  Docs with >1 passage in pool:          "
                f"{self.multi_passage_documents:,} ({multi_pct:.1%})",
                f"  Queries w/ positive siblings in pool:  "
                f"{self.queries_with_positive_siblings_in_pool:,} ({sibling_pct:.1%})",
                f"  Queries w/ same-doc hard negatives:    "
                f"{self.queries_with_same_doc_negatives:,} ({neg_pct:.1%})",
            ]
        )


def analyze_siblings(
    data: ClercSplit,
    pid2did: dict[str, str],
    max_examples: int = 5,
) -> SiblingStats:
    """Trace pooled passages back to documents and measure label leakage.

    Reports how often the pool contains other passages of a query's positive
    document ("right case, wrong slice" is possible), and how often those
    siblings were mined as that query's own hard negatives (the labels then
    actively penalize retrieving the cited case).
    """
    did_to_pids = build_did_to_pids(data.corpus, pid2did)

    queries_with_siblings = 0
    queries_with_same_doc_negatives = 0
    examples: list[SiblingExample] = []
    for qid, pos_pids in data.qrels.items():
        pos_dids = {pid2did[pid] for pid in pos_pids}
        pool_siblings = set().union(*(did_to_pids[did] for did in pos_dids)) - pos_pids
        if pool_siblings:
            queries_with_siblings += 1
        neg_siblings = sorted(
            pid for pid in data.negatives.get(qid, set()) if pid2did[pid] in pos_dids
        )
        if neg_siblings:
            queries_with_same_doc_negatives += 1
            if len(examples) < max_examples:
                examples.append(
                    SiblingExample(
                        query_id=qid,
                        positive_pids=sorted(pos_pids),
                        sibling_negative_pids=neg_siblings,
                    )
                )

    return SiblingStats(
        num_passages=len(data.corpus),
        num_documents=len(did_to_pids),
        multi_passage_documents=sum(
            1 for pids in did_to_pids.values() if len(pids) > 1
        ),
        num_queries=len(data.qrels),
        queries_with_positive_siblings_in_pool=queries_with_siblings,
        queries_with_same_doc_negatives=queries_with_same_doc_negatives,
        examples=examples,
    )
