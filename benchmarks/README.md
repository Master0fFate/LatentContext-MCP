# Retrieval benchmark

Run the representative session-isolated retrieval benchmark from the repository root:

```bash
npm run benchmark:retrieval
```

It creates an isolated database, disables embeddings, and stores 512 deterministic work-item notes: 384 Tier 0 entries and 128 Tier 1 summaries. The notes span eight realistic workstreams, with topic-local lexical overlap and unique checkpoint, owner, and verification fields. This exercises both active retrieval sources without turning all candidates into artificial near-duplicates. The benchmark warms retrieval, then reports mean and p95 latency for 15 bounded (8,000-token) retrievals. Adjust the deterministic workload with `LATENTCONTEXT_BENCH_ENTRIES` and `LATENTCONTEXT_BENCH_RUNS`.

## Measured hot-path change

Deduplication previously constructed two lexical term sets for every candidate pair: `n * (n - 1)` term-set constructions and `n * (n - 1) / 2` similarity checks. Retrieval now constructs each candidate's term set once and uses an inverted term index to aggregate each candidate pair's intersection. It evaluates similarity only for pairs that share a term, without sorting candidate-index lists or re-scanning terms for every pair. For sparse or topic-local overlap this changes pair evaluation from all `O(n²)` pairs to `O(P)`, where `P` is the number of pairs sharing a term; dense overlap remains bounded by the same worst case.

The benchmark emits deterministic before/after work counts alongside timing. At the default 512-entry workload, the index considers 16,128 of the legacy 130,816 pairs (87.67% fewer similarity evaluations) and constructs 512 term sets instead of 261,632. The latency fields are local measurements; rerun on the target machine for timing comparisons.

Tier 0 uses its cached entry token count and Tier 1 uses persisted `token_count` while greedy packing. The final rendered payload is still counted exactly and trailing candidates are removed until it fits, preserving the strict token budget despite BPE boundary effects.
