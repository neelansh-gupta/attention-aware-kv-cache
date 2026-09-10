# Attention-Aware KV Cache Compression

Postman AI/ML Recruitment Task 3 (Batch 25).

This project investigates how different KV-cache eviction strategies affect language-model generation when the KV cache is constrained to a fixed budget.

The central question is:

> When a language model cannot keep every past token in its KV cache, which tokens should be retained to preserve useful context?

The project implements and compares multiple cache-eviction policies with the same underlying language model and cache-budget constraints.

## Model

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

See [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for the detailed implementation status and known limitations.

| Stage | Description                                  | Status      |
| ----- | -------------------------------------------- | ----------- |
| 0     | Repository foundation                        | Complete    |
| 1     | Model loading + KV cache inspection          | Complete    |
| 2     | Attention instrumentation + sink analysis    | Complete    |
| 3     | Sliding-window eviction                      | Complete    |
| 4     | StreamingLLM / attention-sink policy         | Complete    |
| 5     | H2O / heavy-hitter policy                    | Complete    |
| 6     | Correct RoPE / position handling             | Next        |
| 7     | Full correctness harness                     | Not started |
| 8     | Benchmark infrastructure                     | Not started |
| 9     | Perplexity + Needle-in-a-Haystack evaluation | Not started |
| 10    | Visualization + quality-vs-memory curves     | Not started |
| 11    | WRITEUP.md                                   | Not started |
| 12    | Final repository audit                       | Not started |

---

# Implementation

The main components currently include:

```text
src/
├── model_wrapper.py
├── cache_manager.py
└── evictions.py
```

### `src/model_wrapper.py`

Contains:

* Model loading
* Tokenization
* Incremental generation
* Attention inspection
* Attention analysis
* Sliding-window generation
* StreamingLLM / attention-sink-aware generation
* H2O / heavy-hitter generation

### `src/cache_manager.py`

Contains:

* `SlidingWindowCacheManager`
* `AttentionSinkCacheManager`
* `HeavyHitterCacheManager`
* Cache statistics and budget enforcement

### `src/evictions.py`

Contains the cache-eviction utilities used by the different policies, including:

* Sliding-window eviction
* Attention-sink-aware eviction
* Heavy-hitter eviction

---

# Stage 5 Validation

Stage 5 has been validated with unit tests for:

* Heavy-hitter token selection
* Cache-budget enforcement
* Attention-score/cache-length alignment
* No-eviction behavior when the cache is within budget

Run:

```bash
python test_heavy_hitter.py
```

Expected:

```text
Stage 5 heavy-hitter tests passed.
```

Previous-stage regression tests also pass:

```bash
python test_sliding_window.py
python test_attention_sink.py
```

Expected:

```text
Stage 3 sliding-window tests passed.
Stage 4 attention-sink tests passed.
```

The heavy-hitter generation path has also been executed successfully with a small cache budget.

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

Removing tokens from the middle of the cache changes the positional structure seen by subsequent decoding steps.

Correct RoPE / position handling for this situation is therefore intentionally deferred to **Stage 6**.

Consequently, Stage 5 validates the heavy-hitter cache-selection mechanism and its integration into generation, but does **not** claim final long-context generation quality.

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

Run the Stage 5 tests:

```bash
python test_heavy_hitter.py
```

Run the previous-stage regression tests:

```bash
python test_sliding_window.py
python test_attention_sink.py
```

Run the model-wrapper smoke test:

```bash
python -m src.model_wrapper
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
Stage 6   Correct RoPE / position handling              → Next
Stage 7   Full correctness harness
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
│   └── evictions.py
├── results/
├── test_sliding_window.py
├── test_attention_sink.py
├── test_heavy_hitter.py
├── requirements.txt
├── README.md
└── PROJECT_STATUS.md
```

---

# Scope

This project is intentionally developed in stages.

The goal is to implement the cache policies first, validate their mechanics, and then build the infrastructure required for rigorous quality and memory comparisons.

Results will only be reported when they have actually been measured.

No fabricated benchmark, perplexity, retrieval-accuracy, memory, or quality numbers are included.
