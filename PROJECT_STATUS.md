## Stage 6: Correct RoPE / Position Handling after Eviction — COMPLETE

### Goal

Ensure that token eviction does not break positional information for Qwen2's Rotary Position Embeddings (RoPE).

### Implementation

Stage 6 separates the **absolute token position** from the **physical cache position**.

Implemented:

- Added `src/position_utils.py`
- Absolute `position_ids` are maintained across generation.
- `cache_position` continues to represent the absolute position of newly generated tokens.
- Physical attention masks are still trimmed according to the current KV-cache size.
- Added compatibility handling for legacy Transformers RoPE implementations.
- Legacy RoPE cosine/sine caches are extended when the required absolute position exceeds the cached length.
- Updated:
  - `sliding_window_generate`
  - `streaming_llm_generate`
  - `heavy_hitter_generate`

### Tests

All Stage 6 position-handling tests pass:

    Stage 6 position-handling tests passed.

Regression tests also pass:

    Stage 5 heavy-hitter tests passed.
    Stage 4 attention-sink tests passed.
    Stage 3 sliding-window tests passed.

### Generation Validation

Actual generation was tested after the Stage 6 changes:

- Heavy-Hitter / H2O — 16 new tokens: PASS
- Heavy-Hitter / H2O — 64 new tokens: PASS
- Sliding Window — 16 new tokens: PASS
- StreamingLLM — 16 new tokens: PASS

Generation completes without position, RoPE, or cache-position errors.

The H2O generation output remains repetitive at a small cache budget. This is not treated as a Stage 6 quality result; Stage 6 only validates correct positional handling after eviction.

### Known Limitation

Heavy-Hitter eviction remains non-contiguous. Stage 6 ensures that absolute positional information is preserved after such eviction, but generation quality under aggressive KV-cache compression is evaluated separately in later benchmarking stages.

### Status

**Stage 6 complete.**

Next: **Stage 7 — Full Correctness Harness**