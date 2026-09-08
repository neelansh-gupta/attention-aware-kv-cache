## Current Status

- Stage 0: Complete
- Stage 1: Complete
- Stage 2: Complete
- Stage 3: Complete

### Stage 3 — Sliding Window Cache

Implemented a fixed-size sliding-window KV-cache eviction policy.

#### Implementation

- Added sliding-window cache eviction utilities.
- Added `SlidingWindowCacheManager`.
- Integrated the cache manager with model generation.
- Uses Hugging Face's native `DynamicCache`.
- Uses `DynamicCache.crop()` to retain only the most recent `window_size` KV entries.
- Added unit tests for cache eviction and cache-budget enforcement.
- Verified sliding-window generation with the Qwen2.5-0.5B model.

#### Verification

- `python test_sliding_window.py` — passed.
- `python -m src.model_wrapper` — passed.
- Sliding-window generation integration test — passed.

### Next Stage

Stage 4 — StreamingLLM / Attention-Sink-Aware Cache