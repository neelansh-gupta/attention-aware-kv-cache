# Attention-Aware KV Cache Compression

Postman AI/ML Recruitment Task 3 (Batch 25)

## Current status

Repair verification completed against the current repository, live runtime
behavior, and Task 3 in `postman_25.pdf`.

| Stage | Status | Verified evidence |
|---|---|---|
| Stage 0 | PASS | Package/docs/environment foundation present |
| Stage 1 | COMPLETE | Current `DynamicCache` inspection and model smoke |
| Stage 2 | COMPLETE | 256-token measured layer/head/token analysis and plots |
| Stage 3 | COMPLETE | Reference agreement, retained metadata, K/V and generation |
| Stage 4 | COMPLETE | Sink/recent policy tests and real two-policy experiment |
| Stage 5 | COMPLETE | Sinks + recent + accumulated-score heavy hitters |
| Stage 6 | COMPLETE | Real Qwen non-contiguous middle-eviction regression |
| Stage 7 | COMPLETE | Public categorized A/B/C/D correctness entry point |

**Next available stage:** Stage 8, only when explicitly requested.

No Stage 8+ functionality was implemented during this repair. Existing
untracked `benchmark.py` and `results/benchmark_*` predate this repair and are
not used as evidence here.

## Verified environment

```text
Python 3.14.4
torch 2.14.0+cu130
transformers 5.16.1
CUDA available: False
model/device: Qwen/Qwen2.5-0.5B on CPU
```

## Important implementation decisions

- Cache inspection supports current `Cache.layers`, older
  `key_cache`/`value_cache`, and legacy tuple caches without private conversion
  APIs.
- Attention uses eager mode and reports first 1/2/4/8 prefix fractions;
  unavailable prefix sizes remain `null`/unavailable.
- Every manager tracks explicit original `retained_positions`.
- Sliding Window retains the newest K entries.
- StreamingLLM reserves configured sinks and fills the rest with the newest
  local entries.
- H2O priority is deterministic: sinks, recent window, then highest accumulated
  scores; ties favor lower indices; output returns to sequence order.
- Qwen2 applies RoPE before cache insertion. Retained cached keys keep original
  rotations; new tokens receive increasing absolute `position_ids`.
- Post-eviction output equality is not a correctness assertion. Stage 7 records
  generated-token agreement as a descriptive quality measurement.

## Measured Stage 2 result

Command:

```bash
python visualise.py --max-tokens 256 --output-dir results/plots/attention
```

Actual CPU result:

```text
sequence length: 256
layers: 24
heads: 14
most attended token position: 0
first 1 token fraction: 0.3270432464
first 2 token fraction: 0.3363858856
first 4 token fraction: 0.3488528111
first 8 token fraction: 0.3739779924
uniform first-8 positional baseline: 0.03125
```

Artifacts:

```text
results/plots/attention/attention_metrics.json
results/plots/attention/attention_by_token_position.png
results/plots/attention/early_token_attention.png
results/plots/attention/layer_attention_heatmap.png
results/plots/attention/head_attention_heatmap.png
```

This establishes concentration for the measured prompt/model/run; it is not
presented as a universal hard-coded conclusion.

## Stage 4 real experiment

`python stage4_experiment.py` ran Sliding Window and StreamingLLM with the same
model, 34-token prompt, budget 8, and four generated tokens.

Measured retained positions:

```text
Sliding Window: [29, 30, 31, 32, 33, 34, 35, 36]
StreamingLLM:    [0, 1, 31, 32, 33, 34, 35, 36]
```

Both generations completed and respected budget 8. Raw output:
`results/stage4_sliding_vs_streaming.json`.

## Stage 6 verified mechanism

The model-level regression retains original positions
`[0, 1, 2, 7, 8, 9]`. For every Qwen layer, retained K/V exactly equal the
corresponding entries selected from the full cache. The next rotary call
receives absolute position `10`, not shortened-cache position `6`.

The removed `ensure_rope_cache_length` helper was unnecessary for installed
Transformers 5 Qwen2: rotary embeddings are computed from `position_ids`
before `past_key_values.update`.

## Stage 7 categories

`python test_correctness.py` verifies:

- **A — Cache implementation:** full `DynamicCache` vs custom tensor view
  before eviction, with numerical tolerance.
- **B — Eviction:** exact indices, K/V correspondence, explicit metadata,
  score alignment, duplicate prevention, and budgets.
- **C — Position:** actual Qwen middle-token eviction and absolute next
  position.
- **D — Quality:** compressed vs full greedy-token agreement. Equality is not
  required.

Measured four-token agreement in the verified run:

```text
Sliding Window: 0.25
StreamingLLM:   0.25
H2O:            0.25
```

These values are only a correctness-harness smoke measurement, not a benchmark
or general quality claim. Raw output: `results/stage7_quality_comparison.json`.

## Commands used

```bash
python -m src.model_wrapper
python visualise.py --max-tokens 256 --output-dir results/plots/attention
python test_sliding_window.py
python test_attention_sink.py
python stage4_experiment.py
python test_heavy_hitter.py
python test_position_handling.py
python test_correctness_harness.py
python test_correctness.py
```

## Known limitations

- Verification was CPU-only; CUDA-specific behavior was not exercised.
- Stage 2 used a 256-token context due available CPU resources. The metrics are
  prompt-specific.
- The model-level position regression covers installed Transformers 5.16.1
  Qwen2, not every historical/custom RoPE implementation.
- Stage 7's four-token agreement is descriptive smoke data, not perplexity,
  Needle-in-a-Haystack, or a benchmark.
- Perplexity, NIH, benchmark infrastructure, quality-vs-memory curves, and the
  final 2–4 page writeup belong to Stages 8–11 and remain incomplete.

## Phase 1 audit status

```text
Initial Phase 1 audit: AUDITED
Stages 0–7 repair: COMPLETE AND VERIFIED
Stage 0: PASS
Stage 1: COMPLETE
Stage 2: COMPLETE
Stage 3: COMPLETE
Stage 4: COMPLETE
Stage 5: COMPLETE
Stage 6: COMPLETE
Stage 7: COMPLETE
```
