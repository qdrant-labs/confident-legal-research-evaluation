# PLAN — Evaluation Methodology Article + CLERC Implementation

## Goal

Introduce the goal-decomposition evaluation methodology through one concrete,
end-to-end example (legal RAG over CLERC), rather than presenting the full
framework upfront. The article must serve a time-constrained reader: three
artifacts at three depths — a 5-node map (30 sec), one worked slice (10 min),
a reusable dimension table (steal-this template).

**Core narrative device:** setup–payoff. Skip utility early ("hard to compute
— skip for now"), then return to it via UDCG (arXiv 2510.21440) as the climax.

## Article structure

1. **Hook** *(done)* — naive system, real failure (broken-wall query), the
   LLM's fragile safety net. Failure first, framework second.
2. **The map, in miniature** — one paragraph + simplified 5-node diagram of
   the three layers (Decomposition → FR/NFR mapping → Integration). NOT the
   full 40-node framework diagram.
3. **Goal decomposition** *(done, rewritten)* — system-level NFRs
   (Trustworthiness / Correctness / Performance), per-component
   operationalization, depth asymmetry as the finding, "no natural zero" for
   dense search.
4. **Vertical slice: Ranking Quality (search component)**
   - Walk the FR ladder: NFR → "objectively computable?" gate → FR
     ("top-N results should have the biggest relevance") → NDCG@10 →
     golden set = CLERC qrels → run benchmark.
   - Improve the system against NDCG (e.g. better encoder / reranker).
   - Each methodology gate introduced at the moment the example hits it.
5. **Transition artifact** — one short cell: improved top-10 that looks good
   by NDCG but contains near-duplicates / plausible-but-off passages.
   Concrete artifact first, paper second (mirrors the opening rhythm).
6. **The callback: utility ≠ relevance**
   - "Remember 'consumer is LLM — skip for now'? Let's not skip it."
   - Baseline: arXiv 2510.21440 (UDCG — Utility and Distraction-aware
     Cumulative Gain). NDCG's positional discount encodes a *human* consumer;
     our consumer is an LLM that reads the whole context at once.
   - Say explicitly: this is our own principle ("the consumer determines what
     the NFR operationally means") applied one level deeper — the methodology
     *predicts* where the metric breaks.
   - Note the gate crossing: relevance was computable, utility requires a
     judge → demonstrates the NFR/LLM-judge branch without a second full
     walkthrough.
7. **Dimension overview table** — one row per remaining dimension. Columns:
   FR / metric / slice / target / golden-set source / method / cost tier.
   Fill known rows, mark the rest TODO honestly. Full mapping lives in the
   draw.io for interested readers.
8. **Results + the loop** — other evals point to next improvements; the cycle
   is why the Integration layer exists. CI/CD close as a consequence, not a
   moral. One paragraph, pointing at the yellow layer. No full CI section.
9. **Full framework diagram at the end** as reference artifact + honest
   scoping note (branches not exercised: online/AB, full judge validation
   loop, integration gates) → sets up article #2 (RAG-side evaluation, the
   deeper decomposition).

## Implementation work items

### Ranking Quality slice (NDCG)
- [ ] Evaluator: NDCG@10 (+ Recall@K, MRR as secondary) against CLERC qrels
      over `DenseSearcher` results.
- [ ] Baseline run on current naive system (bge-small-en-v1.5, no reranker).
- [ ] One improvement round (candidate: stronger encoder or reranker —
      prefer the change most likely to expose NDCG↑/utility↔ divergence;
      pointwise rerankers are a plausible candidate).
- [ ] Before/after NDCG comparison table.

### Utility slice (UDCG)
- [ ] LLM utility annotator: per retrieved passage, label useful vs
      distracting (utility schema from the paper). Bounded slice — decide
      size upfront (a few hundred queries max; this is the cost-of-
      measurement node in action).
- [ ] UDCG computation over annotated slice.
- [ ] Directional validation only — do NOT reproduce the paper's full
      correlation study. E.g.: UDCG orders our two system variants
      differently than NDCG; spot-check ~50 queries end-to-end to see which
      ordering matches answer quality.
- [ ] Money shot (if it appears): a case where NDCG improves but UDCG /
      end-to-end quality doesn't. Don't force it; plan improvement steps to
      make it likely.

### Supporting
- [ ] Dimension table (section 7) with CLERC-backed rows filled:
      search = Calibration / Ranking Quality / Latency;
      RAG = Factual / User understanding / Form (+ Latency, Cost).
- [ ] Simplified 5-node methodology map (new diagram).
- [ ] Fix full framework diagram before publishing: rename "expressive as
      number?" → "objectively computable from data?"; annotate Failure-Modes
      node ("subsystem interaction / partial-failure combinations");
      typos (Assisstant → Assistant, Explanable → Explainable,
      Dimensionbs → Dimensions); rename RAG "Non-factual" bucket → "Form"
      (move Grounded/Traceable/Confidence-calibrated under Factual).

## Scope decisions (settled)

- **One vertical slice, not eight.** The table proves coverage; the slice
  proves the method. Seven walkthroughs = padding.
- **System is atomic, not distributed** — search + RAG deploy together, so
  the failure-modes branch does not apply. (Fault-injection/robustness
  testing is a different concept; out of scope, possibly RAG-side later.)
- **Search trustworthiness = Calibration** (single sub-requirement at PoC
  stage; reliability deferred). Dense search has no natural zero — calibrated
  emptiness must be engineered via threshold τ, measured on a negatives
  slice. Overview-table row only in this article.
- **Do not improve the system against UDCG** — show the measurement
  discrepancy, leave utility-driven improvement as the sequel hook.
- **CLERC train split, limit=2000** → corpus ≈ 41k docs, embedded + indexed
  in Qdrant (`lawyer_citations`, bge-small-en-v1.5, 384-dim, cosine).

## Current repo state

- `src/dataset.py` — CLERC loader, dict-keyed corpus/queries, qrels, sample().
- `src/indexing.py` — BaseIndexer/DocumentIndexer, EmbeddingConfig
  (providers + parallel), disk-backed EmbeddingCache, chunked upsert.
- `src/search.py` — DenseSearcher (query API only, no reranker).
- `src/rag.py` — LegalAssistant (instructor + litellm, claude-sonnet-4-6),
  Citation/RAGAnswer models.
- `experiments.ipynb` — narrative through goal decomposition (rewritten);
  next section = Ranking Quality slice.
- `diagrams/` — eval-inputs, system-goal-decomposition,
  component-level-goal-decomposition (drawio SVGs).
