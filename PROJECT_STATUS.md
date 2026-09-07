# Project Status

## Completed stages

- **Stage 0 — Repository Foundation**: `src/` package skeleton, `requirements.txt`,
  `.gitignore`, `README.md`, this file. No KV compression logic yet.

## Current stage

Stage 0 (finishing up).

## Next stage

Stage 1 — Model Loading + KV Cache Inspection (`src/model_wrapper.py`):
load Qwen2.5-0.5B, auto device selection, forward pass, incremental
generation, inspect `past_key_values` shapes, report layer/head/KV
dimensions, expose attention scores where supported.

## Tests completed

None yet (no runnable functionality beyond package import).

Verified for Stage 0:
- `python -c "import src"` succeeds with no errors from a fresh checkout.

## Known issues

- None yet. Repository has no functional code beyond package scaffolding.

## Important implementation decisions

- **Model**: `Qwen/Qwen2.5-0.5B`, chosen per task spec — small enough for
  CPU/laptop/Colab-T4 execution.
- **Dependency pinning**: `transformers>=4.40.0,<5.0.0` to give room for a
  version investigation in Stage 1 (Qwen2.5's `past_key_values` behavior
  differs between legacy tuple-based caches and the newer HF `Cache`
  object API — this must be checked against the actually-installed version
  rather than assumed).
- **`.gitignore`** excludes large/regenerable artifacts (`results/raw/`,
  model weight files, tensor dumps) so the repo stays lightweight; plots and
  small JSON/CSV result summaries are expected to be committed once they
  exist.

## Commands used to verify current stage

```bash
python -c "import src; print(src.__version__)"
```
Expected output: `0.0.1`
