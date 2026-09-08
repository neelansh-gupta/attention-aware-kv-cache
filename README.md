# Attention-Aware KV Cache Compression

Postman AI/ML Recruitment Task 3 (Batch 25).

This project investigates how compressing a Transformer KV cache under a fixed memory budget affects generation quality.

The central question is:

> How can we reduce the memory required by a language model's KV cache without unnecessarily destroying information that is important for generation?

The project implements multiple KV-cache eviction policies and will eventually compare their effect on generation quality under the same memory budget.

The three policies are:

1. **Sliding Window** — naive recency-only baseline. Retains only the most recent tokens and serves as the baseline for comparison.
2. **Attention-Sink-Aware (StreamingLLM-style)** — keeps a small number of initial "sink" tokens together with a recent local window. This is motivated by the observation that early tokens can receive disproportionately large attention.
3. **Accumulated-Score / Heavy-Hitter (H2O-style)** — retains tokens based on historically accumulated attention mass. This policy will be implemented in Stage 5.

## Model

The project uses:

* `Qwen/Qwen2.5-0.5B`
* The model is small enough to run on CPU or on a free Colab T4 / laptop GPU.
* A CUDA GPU is not required for the current implementation.

## Project Status

See `PROJECT_STATUS.md` for the authoritative implementation status, verification results, current scope, and next stage.

This project is built incrementally in stages. Do not assume anything beyond what `PROJECT_STATUS.md` reports as complete.

### Completed

| Stage | Description | Status |
|---|---|---|
| 0 | Repository foundation | Complete |
| 1 | Model loading + KV cache inspection | Complete |
| 2 | Attention instrumentation + sink analysis | Complete |
| 3 | Sliding window eviction | Complete |
| 4 | StreamingLLM / attention-sink policy | Complete |

### Upcoming

| Stage | Description | Status |
|---|---|---|
| 5 | H2O / heavy-hitter policy | Next |
| 6 | Correct RoPE / position handling after eviction | Planned |
| 7 | Full correctness harness | Planned |
| 8 | Benchmark infrastructure | Planned |
| 9 | Perplexity + Needle-in-a-Haystack evaluation | Planned |
| 10 | Visualization + quality-vs-memory curves | Planned |
| 11 | WRITEUP.md | Planned |
| 12 | Final repository audit | Planned |

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

* Requires Python 3.10+.
* A CUDA GPU is not required. The code automatically selects CPU or CUDA when available.
* Attention instrumentation and longer-context experiments may be slow on CPU.

## Execution

Execution entry points and tests are added stage by stage.

### Model loading, generation, and attention analysis

Run:

```bash
python -m src.model_wrapper
```

This currently verifies:

* `Qwen2.5-0.5B` model loading
* Device selection
* Incremental generation
* KV-cache usage
* Attention extraction
* Attention-sink analysis

Example output includes:

```text
Model: Qwen/Qwen2.5-0.5B
Device: cpu

Incremental generation:
...

Attention analysis:
Most attended token index: 0
...

Attention sink analysis:
Early-token attention fraction: 0.97265625
```

The attention-sink analysis provides motivation for the attention-sink-aware cache policy implemented in Stage 4.

### Stage 3 — Sliding Window Cache

Stage 3 implements the naive recency-only KV-cache baseline. The policy retains only the most recent tokens within a fixed cache budget.

For example:

```text
Original cache:
[0 1 2 3 4 5 6 7 8 9]

Budget = 6

Retained:
[4 5 6 7 8 9]
```

The implementation uses Hugging Face's native `DynamicCache` and provides a `SlidingWindowCacheManager`.

Run the Stage 3 tests:

```bash
python test_sliding_window.py
```

Expected:
`Stage 3 sliding-window tests passed.`

Stage 3 also supports sliding-window generation through the model wrapper.

### Stage 4 — StreamingLLM / Attention-Sink-Aware Cache

Stage 4 implements a StreamingLLM-style KV-cache retention policy. Instead of keeping only the most recent tokens, the policy preserves:

* a fixed number of initial attention-sink tokens
* the most recent tokens using the remaining cache budget

For a cache budget $B$ and $S$ sink tokens:

```text
Original cache:
[0 1 2 3 4 5 6 7 8 9]

B = 6
S = 2

Retained:
[0 1 | 6 7 8 9]
  ↑       ↑
sinks   recent tokens
```

In other words: **first S tokens + latest (B - S) tokens** are retained.

#### Why preserve the initial tokens?

Stage 2 attention analysis showed that early tokens can receive a very large fraction of attention. For example, the current Stage 2 analysis observed:

```text
Most attended token index: 0
Early-token attention fraction: 0.97265625
```

Stage 4 therefore tests a simple hypothesis: **If some initial tokens behave as attention sinks, preserving them may be preferable to discarding them under a fixed cache budget.**

#### Stage 3 vs Stage 4

**Sliding Window:**
```text
[0 1 2 3 4 5 6 7 8 9]
              └───────┘
             recent only
```

**Attention-Sink-Aware:**
```text
[0 1 2 3 4 5 6 7 8 9]
 └─┘           └───────┘
sinks          recent
```

The two policies use the same basic cache-budget constraint but retain different parts of the sequence.

#### Stage 4 tests

Run:

```bash
python test_attention_sink.py
```

Expected:
`Stage 4 attention-sink tests passed.`

Stage 3 should also continue to pass:

```bash
python test_sliding_window.py
```

#### Stage 4 generation

The StreamingLLM-style generation path can be tested with:

```bash
python -c "from src.model_wrapper import ModelWrapper; m=ModelWrapper(); print(m.streaming_llm_generate('The history of artificial intelligence is', max_new_tokens=16, cache_budget=8, sink_tokens=2))"
```

This verifies that `Qwen2.5-0.5B` can generate using the attention-sink-aware cache manager.

## Current Architecture

The current cache architecture is:

```text
                         Model
                           │
                           ▼
                      DynamicCache
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
      Sliding Window              Attention Sink
       Cache Manager               Cache Manager
             │                           │
             ▼                           ▼
      Recent tokens              Sink + recent tokens
```

The two policies can therefore be compared while enforcing a fixed cache budget.

## Repository Structure

```text
ai-ml/
├── src/
│   ├── __init__.py
│   ├── model_wrapper.py
│   ├── cache_manager.py
│   └── evictions.py
│
├── test_sliding_window.py
├── test_attention_sink.py
├── test_correctness.py
├── benchmark.py
├── visualize.py
│
├── requirements.txt
├── README.md
├── PROJECT_STATUS.md
├── WRITEUP.md
│
└── results/
    └── plots/
```

*Some files shown above are reserved for later stages and may not yet contain their final implementation.*

## Design Scope

The project is intentionally developed in stages so that each cache policy and supporting component can be implemented and verified independently.

* **Stage 0:** Repository foundation and project structure.
* **Stage 1:** Model loading, tokenization, incremental generation, and KV-cache inspection.
* **Stage 2:** Attention extraction, attention aggregation, and attention-sink analysis.
* **Stage 3:** Sliding-window KV-cache eviction under a fixed cache budget.
* **Stage 4:** StreamingLLM / attention-sink-aware KV-cache eviction.
* **Stage 5:** H2O / heavy-hitter cache policy.
* **Stage 6:** Correct RoPE and position handling after non-contiguous cache eviction.

Later stages will add correctness testing, benchmarking, quality evaluation, visualization, and the final write-up.

### Important Scope Note

Stage 4 currently verifies the cache policy and generation path. It does not yet establish that the attention-sink-aware policy produces better generation quality than the sliding-window baseline.

No claims are currently made about:

* perplexity improvement
* Needle-in-a-Haystack accuracy
* generation-quality improvement
* memory reduction beyond the configured cache budget
* benchmark performance

Those quantitative comparisons belong to the later evaluation stages. Correct RoPE / position handling after non-contiguous cache eviction is also intentionally deferred to Stage 6.

## Verification

The following Stage 4 verification has been completed:

* `python test_attention_sink.py` → Stage 4 attention-sink tests passed.
* `python test_sliding_window.py` → Stage 3 sliding-window tests passed.
* `python -m src.model_wrapper` → Model loading, generation, attention analysis, and sink analysis passed.

### StreamingLLM generation integration test

Generation completed successfully. The Stage 4 generation path was tested with:

* **Model:** `Qwen/Qwen2.5-0.5B`
* **Cache budget:** 8
* **Sink tokens:** 2
* **New tokens:** 16

The model successfully generated text using the attention-sink-aware cache policy.

## Roadmap

| Stage | Task | Status |
|---|---|---|
| Stage 0 | Repository foundation | ✅ |
| Stage 1 | Model + KV-cache inspection | ✅ |
| Stage 2 | Attention instrumentation + sink analysis | ✅ |
| Stage 3 | Sliding-window cache | ✅ |
| Stage 4 | StreamingLLM / attention-sink cache | ✅ |
| Stage 5 | H2O / heavy-hitter policy | ⏳ |
| Stage 6 | RoPE / position handling | ⏳ |
| Stage 7 | Correctness harness | ⏳ |
| Stage 8 | Benchmark infrastructure | ⏳ |
| Stage 9 | Perplexity + Needle-in-a-Haystack evaluation | ⏳ |
| Stage 10 | Visualization + quality/memory curves | ⏳ |
| Stage 11 | WRITEUP.md | ⏳ |
| Stage 12 | Final repository audit | ⏳ |

## Next Stage

### Stage 5 — H2O / Heavy-Hitter Policy

The next policy will retain tokens based on accumulated attention mass. The eventual comparison will be:

$$\text{Sliding Window} \longrightarrow \text{Attention Sink / StreamingLLM} \longrightarrow \text{H2O / Heavy-Hitter}$$

under the same cache budget. The goal is to determine whether attention-aware eviction policies retain more useful information than a purely recency-based policy.

## Reproducibility and Reporting

Per the task brief, incomplete submissions are expected. This repository reports honestly which stages, tests, and experiments have actually been completed.

No fabricated:

* perplexity numbers
* Needle-in-a-Haystack accuracy
* memory measurements
* benchmark results
* generation-quality improvements

are reported. Quantitative evaluation will be added in the later benchmarking and evaluation stages.