# Project Status

## Completed Stages

- Stage 0 — Repository Foundation
- Stage 1 — Model Loading + KV Cache Inspection

## Current Stage

Stage 1 completed.

## Next Stage

Stage 2 — Attention Instrumentation + Attention Sink Analysis

## Stage 1 Implementation

Implemented:

- Qwen/Qwen2.5-0.5B model loading.
- Tokenizer loading.
- Automatic CPU/CUDA device selection.
- Normal forward pass.
- Incremental token-by-token generation.
- KV-cache inspection.
- Compatibility with modern Hugging Face Cache objects.
- Compatibility with legacy tuple-style `past_key_values`.
- Attention tensor inspection.
- Stage 1 smoke test through `python -m src.model_wrapper`.

No KV-cache eviction has been implemented.

## Tests Completed

Stage 1 smoke test:

```bash
python -m src.model_wrapper
```

Expected output: `0.0.2`
