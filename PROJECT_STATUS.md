# Project Status

## Current Status

* Stage 0: Complete
* Stage 1: Complete
* Stage 2: Complete
* Stage 3: Complete
* Stage 4: Complete
* Stage 5: Complete

---

## Stage 5 — H2O / Heavy-Hitter KV-Cache Policy

Implemented an H2O-style accumulated-attention KV-cache eviction policy that retains tokens based on the attention mass they have historically received, rather than retaining only recent tokens.

### Implementation

* Added `heavy_hitter_evict()` for score-based KV-cache eviction.
* Added `HeavyHitterCacheManager`.
* Tracks accumulated attention scores for cached tokens.
* Selects the highest-scoring tokens when the cache exceeds the configured budget.
* Preserves the original sequence order of the retained tokens.
* Updates the accumulated attention scores after eviction.
* Supports legacy tuple-style KV caches.
* Supports modern Hugging Face cache representations.
* Added `heavy_hitter_generate()` to `ModelWrapper`.
* Enables attention outputs during generation for heavy-hitter scoring.
* Added Stage 5 unit tests covering token selection, cache-budget enforcement, score alignment, and no-eviction behavior.
* Preserved the existing Stage 3 sliding-window implementation.
* Preserved the existing Stage 4 StreamingLLM / attention-sink implementation.
* Verified Stage 3 regression tests pass.
* Verified Stage 4 regression tests pass.
* Verified Stage 5 heavy-hitter tests pass.
* Verified heavy-hitter generation executes successfully with a small cache budget.

### Heavy-Hitter Policy

For a cache budget `B`, each cached token receives an accumulated attention score representing the attention mass it has received during generation.

When the cache exceeds the budget:

1. Rank cached tokens by accumulated attention score.
2. Select the top `B` tokens.
3. Retain their KV entries.
4. Preserve their original sequence order.
5. Continue generation using the reduced cache.

Example:

```text
Original tokens:
[0 1 2 3 4 5 6 7 8 9]

Accumulated attention:
[0.2 0.9 0.1 0.8 0.3 0.7 0.4 0.1 0.6 0.2]

Cache budget:
B = 5

Highest-scoring tokens:
[1 3 5 6 8]

Retained in original sequence order:
[1 3 5 6 8]
```

### Validation

The following tests pass:

```text
python test_heavy_hitter.py
Stage 5 heavy-hitter tests passed.

python test_sliding_window.py
Stage 3 sliding-window tests passed.

python test_attention_sink.py
Stage 4 attention-sink tests passed.
```

The heavy-hitter generation path was also executed successfully with:

```text
cache_budget = 8
max_new_tokens = 64
```

Generation completes successfully, although the very small cache budget produces highly repetitive output. This is not treated as a quality benchmark or final evaluation result.

### Known Limitation

Heavy-hitter eviction is non-contiguous: tokens can be removed from the middle of the original sequence.

Correct RoPE/position handling after non-contiguous KV-cache eviction has **not** yet been implemented and remains deferred to Stage 6.

Therefore, Stage 5 establishes the heavy-hitter cache-selection and eviction mechanism, but does not claim final generation-quality correctness after non-contiguous eviction.

---

## Stage 6 — Correct RoPE / Position Handling

Next stage.

The goal is to correctly handle positional information when KV-cache entries are removed non-contiguously by the heavy-hitter policy.

No Stage 6 implementation or quality claims have been made yet.

---

## Remaining Roadmap

| Stage | Description                                     | Status      |
| ----- | ----------------------------------------------- | ----------- |
| 0     | Repository foundation                           | Complete    |
| 1     | Model loading + KV cache inspection             | Complete    |
| 2     | Attention instrumentation + sink analysis       | Complete    |
| 3     | Sliding-window eviction                         | Complete    |
| 4     | StreamingLLM / attention-sink policy            | Complete    |
| 5     | H2O / heavy-hitter policy                       | Complete    |
| 6     | Correct RoPE / position handling after eviction | Next        |
| 7     | Full correctness harness                        | Not started |
| 8     | Benchmark infrastructure                        | Not started |
| 9     | Perplexity + Needle-in-a-Haystack evaluation    | Not started |
| 10    | Visualization + quality-vs-memory curves        | Not started |
| 11    | WRITEUP.md                                      | Not started |
| 12    | Final repository audit                          | Not started |

---

## Scope

The project is being developed incrementally. Each stage is validated independently before moving to the next stage.

No fabricated perplexity, Needle-in-a-Haystack accuracy, memory, benchmark, or quality numbers are reported. Such evaluations will be added only after the corresponding evaluation infrastructure is implemented.
