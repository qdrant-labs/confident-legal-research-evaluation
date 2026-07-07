import logging
import random

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from tqdm.auto import tqdm
from typing_extensions import override

logger = logging.getLogger(__name__)

REPO_ID = "jhu-clsp/CLERC"


class Document(BaseModel):
    doc_id: str
    text: str


class Query(BaseModel):
    query_id: str
    query: str


class ClercSplit(BaseModel):
    """Loaded CLERC split: deduped corpus, queries, and qrels.

    - corpus: passages unioned over positives + negatives, deduped by doc_id.
    - queries: query rows from the source split.
    - qrels: query_id -> set of relevant doc_ids (positives only).
    - negatives: query_id -> doc_ids of the hard negatives mined for that query.
    """

    corpus: dict[str, Document]
    queries: dict[str, Query]
    qrels: dict[str, set[str]]
    negatives: dict[str, set[str]]

    @override
    def __repr__(self) -> str:
        return (
            f"ClercSplit(corpus={len(self.corpus)} docs, "
            f"queries={len(self.queries)}, qrels={len(self.qrels)})"
        )

    def sample(self) -> tuple[Query, Document]:
        qid = random.choice(list(self.qrels))
        doc_id = random.choice(list(self.qrels[qid]))
        return self.queries[qid], self.corpus[doc_id]


def load_clerc(
    split: str = "test",
    limit: int | None = None,
    name: str | None = None,
) -> ClercSplit:
    """Load the first `limit` rows of a CLERC split and derive corpus + qrels.

    Each source row has positive_passages (relevant — used for qrels) and
    negative_passages (hard distractors). Both are pooled into the corpus and
    deduped by doc_id so the same passage is indexed once even when it appears
    as positive for one query and negative for another.
    """
    ds = load_dataset(REPO_ID, name=name, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    corpus: dict[str, Document] = {}
    queries: dict[str, Query] = {}
    qrels: dict[str, set[str]] = {}
    negatives: dict[str, set[str]] = {}

    for row in tqdm(ds, desc="Dataset to pydantic transformation", total=len(ds)):
        if row is None or not isinstance(row, dict):
            logger.warning(f"Cannot process the row that is not a dictionary: {row}")
            continue
        qid = row["query_id"]
        queries[qid] = Query(query_id=qid, query=row["query"])
        qrels[qid] = {p["docid"] for p in row["positive_passages"]}
        negatives[qid] = {p["docid"] for p in row["negative_passages"]}
        for p in (*row["positive_passages"], *row["negative_passages"]):
            did = p["docid"]
            if did not in corpus:
                corpus[did] = Document(doc_id=did, text=p["text"])

    return ClercSplit(corpus=corpus, queries=queries, qrels=qrels, negatives=negatives)


def download_collection(local_dir: str = "./clerc") -> str:
    """Download the full ~8M-doc CLERC corpus TSV (~8GB gzipped)."""
    return hf_hub_download(
        repo_id=REPO_ID,
        filename="collection/collection.doc.tsv.gz",
        repo_type="dataset",
        local_dir=local_dir,
    )
