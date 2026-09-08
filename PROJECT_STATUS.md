# Project Status

## Current Status

- Stage 0: Complete
- Stage 1: Complete
- Stage 2: Complete
- Stage 3: Complete
- Stage 4: Complete

---

## Stage 4 — StreamingLLM / Attention-Sink-Aware Cache

Implemented a StreamingLLM-style KV-cache eviction policy that preserves
initial attention-sink tokens while retaining a recent local window under
a fixed cache budget.

### Implementation

- Added attention-sink-aware KV-cache eviction utilities.
- Added `AttentionSinkCacheManager`.
- Preserves a fixed number of initial sink tokens.
- Retains the most recent tokens using the remaining cache budget.
- Added StreamingLLM-style generation to `ModelWrapper`.
- Supports the native Hugging Face cache representation.
- Added synthetic tests for prefix/suffix retention.
- Added cache-budget enforcement tests.
- Preserved the existing Stage 3 sliding-window implementation.
- Correct RoPE/position handling after eviction remains deferred to Stage 6.

### Cache Policy

For a cache budget `B` and `S` sink tokens:

```text
Original cache:
[0 1 2 3 4 5 6 7 8 9]

B = 6
S = 2

Retained:
[0 1 | 6 7 8 9]
  ↑       ↑
sinks   recent tokens