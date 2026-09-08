# Project Status

## Completed Stages
- Stage 0 — Repository Foundation
- Stage 1 — Model Loading + KV Cache Inspection
- Stage 2 — Attention Instrumentation + Attention Sink Analysis

## Current Stage
Stage 2 completed.

## Next Stage
Stage 3 — Sliding Window Cache

## Stage 2 Implementation
Implemented:
- Attention extraction from model forward passes.
- Per-layer attention tensor inspection.
- Attention averaging across heads.
- Attention received by each token position.
- Early-token attention analysis for the first 1, 2, 4, and 8 tokens.
- Most-attended token position measurement.
- Attention distribution visualization.
- Early-token attention visualization.
- Layer-wise attention visualization.
- First-layer head-wise attention visualization.
- Experimental analysis without hard-coding the presence of attention sinks.
- Attention analysis output stored under `results/attention/`.

No KV-cache eviction policy has been implemented yet.

## Tests Completed
Stage 2 attention analysis:

```bash
python visualize.py
```
