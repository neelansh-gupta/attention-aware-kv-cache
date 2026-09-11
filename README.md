# Attention-Aware KV Cache Compression

Postman AI/ML Recruitment Task 3 (Batch 25).

This project investigates how different KV-cache eviction strategies affect language-model generation when the KV cache is constrained to a fixed budget.

The central question is:

> When a language model cannot keep every past token in its KV cache, which tokens should be retained to preserve useful context?

The project implements and compares multiple cache-eviction policies with the same underlying language model and cache-budget constraints.

---

# Model

The experiments use:

```text
Qwen/Qwen2.5-0.5B
```

The model is small enough to run on CPU or a free Colab T4 / laptop GPU.

---

# Cache Policies

The project currently implements three KV-cache eviction policies.

### 1. Sliding Window

A recency-only baseline.

When the cache exceeds the configured budget, the oldest tokens are removed and only the most recent tokens are retained.

```text
[0 1 2 3 4 5 6 7 8 9]

Budget = 6

Retained:

[4 5 6 7 8 9]
```

This provides a simple baseline for comparing more informed eviction strategies.

### 2. Attention-Sink-Aware / StreamingLLM

Preserves a small number of initial tokens that tend to receive disproportionate attention, while using the remaining cache budget for the most recent tokens.

```text
[0 1 2 3 4 5 6 7 8 9]

Budget = 6
Sink tokens = 2

Retained:

[0 1 | 6 7 8 9]
  ↑       ↑
 sinks   recent
```

### 3. H2O / Heavy-Hitter

Retains tokens according to accumulated attention scores.

Instead of assuming that recent tokens are the most useful, the policy tracks how much attention each cached token has historically received.

```text
Tokens:

[0 1 2 3 4 5 6 7 8 9]

Accumulated attention:

[.2 .9 .1 .8 .3 .7 .4 .1 .6 .2]

Budget = 5

Selected:

[1 3 5 6 8]
```

The selected tokens are retained in their original sequence order.

---

# Current Project Status

See [`PROJECT_STATUS.md`](https://chatgpt.com/c/PROJECT_STATUS.md) for the detailed implementation status and known limitations.

| Stage | Description | Status |
| ----- | ----------- | ------ |
| 0 | Repository foundation | Complete |
| 1 | Model loading + KV cache inspection | Complete |
| 2 | Attention instrumentation + sink analysis | Complete |
| 3 | Sliding-window eviction | Complete |
| 4 | StreamingLLM / attention-sink policy | Complete |
| 5 | H2O / heavy-hitter policy | Complete |
| 6 | Correct RoPE / position handling | Complete |
| 7 | Full correctness harness | Not started |
| 8 | Benchmark infrastructure | Not started |
| 9 | Perplexity + Needle-in-a-Haystack evaluation | Not started |
| 10 | Visualization + quality-vs-memory curves | Not started |
| 11 | WRITEUP.md | Not started |
| 12 | Final repository audit | Not started |

---

# Implementation

The main components currently include:

```text
src/

├── model_wrapper.py
├── cache_manager.py
├── evictions.py
└── position_utils.py
```

### `src/model_wrapper.py`

Contains:

- Model loading
- Tokenization
- Incremental generation
- Attention inspection
- Attention analysis
- Sliding-window generation
- StreamingLLM / attention-sink-aware generation
- H2O / heavy-hitter generation
- Absolute position tracking during decoding
- RoPE compatibility handling after cache eviction

### `src/cache_manager.py`

Contains:

- `SlidingWindowCacheManager`
- `AttentionSinkCacheManager`
- `HeavyHitterCacheManager`
- Cache statistics and budget enforcement
- Attention-score tracking for heavy-hitter selection

### `src/evictions.py`

Contains the cache-eviction utilities used by the different policies, including:

- Sliding-window eviction
- Attention-sink-aware eviction
- Heavy-hitter eviction

### `src/position_utils.py`

Contains Stage 6 position-handling utilities:

- Absolute `position_ids` construction
- Absolute `cache_position` construction
- Legacy RoPE cache extension

The position utilities keep the model's absolute token positions separate from the physical number of entries currently stored in the KV cache.

---

# Stage 6 — Correct RoPE / Position Handling after Eviction

Stage 6 addresses positional correctness when KV-cache entries are removed during generation.

This is particularly important for H2O / heavy-hitter eviction because the retained cache entries can be non-contiguous.

For example:

```text
Original token positions:

[0 1 2 3 4 5 6 7]

After eviction:

[0 2 4 7]
```

The physical cache now contains four entries, but those entries still correspond to absolute sequence positions `0, 2, 4, 7`.

Stage 6 therefore separates:

- **absolute token position**
- **physical KV-cache length**

### Position Handling

Generation now tracks absolute positions independently of the current cache size.

For example:

```text
Original sequence:

positions = [0 1 2 3 4 5 6 7]

After eviction:

cache contains positions = [0 2 4 7]

Next generated token:

position_id = 8
cache_position = 8
```

The physical cache may contain fewer entries, but the newly generated token still receives its correct absolute position.

### RoPE Compatibility

Legacy Transformers implementations may maintain an internal cosine/sine cache for RoPE.

Stage 6 adds compatibility handling that extends this cache when the required absolute position exceeds the currently cached range.

This prevents absolute position IDs from becoming invalid simply because the physical KV cache has been compressed.

### Updated Generation Paths

Stage 6 position handling is applied to:

- Sliding Window generation
- StreamingLLM / attention-sink generation
- H2O / heavy-hitter generation

The physical attention mask continues to match the actual cache length while absolute position information is maintained separately.

---

# Stage 6 Validation

Dedicated position-handling tests were added in:

```text
test_position_handling.py
```

Run:

```bash
python test_position_handling.py
```

Expected:

```text
Stage 6 position-handling tests passed.
```

The tests validate:

- Absolute position IDs continuing after eviction
- Absolute cache positions
- Legacy RoPE cache extension
- Avoiding unnecessary RoPE cache rebuilding

### Previous-Stage Regression Tests

The Stage 6 changes were also tested against all previous eviction policies:

```bash
python test_heavy_hitter.py
python test_attention_sink.py
python test_sliding_window.py
```

Expected:

```text
Stage 5 heavy-hitter tests passed.
Stage 4 attention-sink tests passed.
Stage 3 sliding-window tests passed.
```

### Actual Generation Validation

Generation was also executed after the Stage 6 changes.

Validated configurations:

```text
Heavy-Hitter / H2O
    cache budget = 8
    generated tokens = 16
    Result: PASS

Heavy-Hitter / H2O
    cache budget = 8
    generated tokens = 64
    Result: PASS

Sliding Window
    window size = 8
    generated tokens = 16
    Result: PASS

StreamingLLM
    cache budget = 8
    sink tokens = 2
    generated tokens = 16
    Result: PASS
```

All four generation paths completed without position, RoPE, or cache-position errors.

The H2O output can still become repetitive under a very small cache budget. This is not considered a Stage 6 quality result. Stage 6 validates positional handling; generation quality is evaluated separately in later benchmark stages.

---

# Important Limitation

The H2O / heavy-hitter policy performs **non-contiguous KV-cache eviction**.

For example:

```text
Original:

[0 1 2 3 4 5 6 7]

Heavy hitters:

[0 2 4 7]
```

Removing tokens from the middle of the cache changes the physical structure of the cached sequence.

Stage 6 addresses the positional side of this problem by preserving the absolute positions associated with generated tokens.

However, correct position handling does not imply that aggressive cache compression will preserve generation quality.

Therefore:

- Stage 5 validates heavy-hitter cache selection and integration.
- Stage 6 validates position/RoPE handling after eviction.
- No final generation-quality conclusions are drawn yet.

No perplexity, Needle-in-a-Haystack accuracy, memory, or benchmark numbers are reported until the relevant evaluation infrastructure is implemented.

---

# Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Python 3.10+ is required.

A CUDA GPU is not required. The model wrapper automatically selects CPU, CUDA, or MPS when available.

---

# Testing

### Stage 6 Position Tests

```bash
python test_position_handling.py
```

### Heavy-Hitter Tests

```bash
python test_heavy_hitter.py
```

### Attention-Sink Tests

```bash
python test_attention_sink.py
```

### Sliding-Window Tests

```bash
python test_sliding_window.py
```

### Model-Wrapper Smoke Test

```bash
python -m src.model_wrapper
```

---

# Example Generation

### Sliding Window

```python
from src.model_wrapper import ModelWrapper

model = ModelWrapper()

output = model.sliding_window_generate(
    "The history of artificial intelligence is",
    max_new_tokens=16,
    window_size=8,
)

print(output)
```

### StreamingLLM / Attention Sink

```python
from src.model_wrapper import ModelWrapper

model = ModelWrapper()

output = model.streaming_llm_generate(
    "The history of artificial intelligence is",
    max_new_tokens=16,
    cache_budget=8,
    sink_tokens=2,
)

print(output)
```

### H2O / Heavy-Hitter

```python
from src.model_wrapper import ModelWrapper

model = ModelWrapper()

output = model.heavy_hitter_generate(
    "The history of artificial intelligence is",
    max_new_tokens=16,
    cache_budget=8,
)

print(output)
```

---

# Roadmap

The project is being developed incrementally:

```text
Stage 0   Repository foundation                         ✓
Stage 1   Model + KV-cache inspection                   ✓
Stage 2   Attention instrumentation + sink analysis     ✓
Stage 3   Sliding-window eviction                       ✓
Stage 4   StreamingLLM / attention-sink policy          ✓
Stage 5   H2O / heavy-hitter policy                     ✓
Stage 6   Correct RoPE / position handling              ✓
Stage 7   Full correctness harness                      → Next
Stage 8   Benchmark infrastructure
Stage 9   Perplexity + Needle-in-a-Haystack evaluation
Stage 10  Visualization + quality-vs-memory curves
Stage 11  WRITEUP.md
Stage 12  Final repository audit
```

---

# Repository Structure

```text
ai-ml/

├── src/
│   ├── __init__.py
│   ├── model_wrapper.py
│   ├── cache_manager.py
│   ├── evictions.py
│   └── position_utils.py
│
├── results/
│
├── test_sliding_window.py
├── test_attention_sink.py
├── test_heavy_hitter.py
├── test_position_handling.py
│
├── requirements.txt
├── README.md
└── PROJECT_STATUS.md
```

---

# Scope

This project is intentionally developed in stages.

The goal is to:

1. Understand KV-cache behavior.
2. Implement multiple cache-eviction policies.
3. Validate their mechanics independently.
4. Correct positional handling after eviction.
5. Build a rigorous correctness and benchmarking harness.
6. Measure generation quality and memory trade-offs.
7. Compare the policies using reproducible experiments.

Results will only be reported when they have actually been measured.

No fabricated benchmark, perplexity, retrieval-accuracy, memory, or quality numbers are included.

---

# Current Status

Stages 0–6 are complete.

The next step is **Stage 7: Full Correctness Harness**, which will systematically validate cache behavior, generation consistency, budget enforcement, and edge cases before benchmark infrastructure is introduced.