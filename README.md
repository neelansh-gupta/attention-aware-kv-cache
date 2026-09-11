# Attention-Aware KV Cache Compression

Postman AI/ML Recruitment Task 3 (Batch 25).

This project investigates why naively dropping old tokens from a language model's KV cache can hurt generation, and compares different KV cache compression strategies that attempt to preserve the most useful information.

The project is implemented around a small Hugging Face causal language model and focuses on understanding, implementing, and validating different cache eviction policies.

---

## Project Goal

During autoregressive generation, a transformer stores previously computed keys and values in a **KV cache**.

The cache grows with every generated token.

A larger KV cache:

- Requires more memory
- Increases attention computation
- Can increase generation latency

A simple solution is to remove old tokens.

However, not all tokens are equally important.

This project investigates three different approaches:

1. **Sliding Window**
2. **StreamingLLM-style Attention Sinks**
3. **H2O-style Heavy Hitters**

The project also addresses an important issue that appears after cache eviction:

> The physical KV cache becomes shorter, but the model's token positions must still remain correct.

Stage 6 therefore adds explicit position handling for RoPE-based models.

Stage 7 adds a deterministic correctness harness to verify the cache implementations before quantitative benchmarking.

---

# Measured Attention Instrumentation

Run:

    python visualise.py --max-tokens 256 --output-dir results/plots/attention

The verified CPU run measured 256 tokens across all 24 layers and 14 attention
heads. It saved JSON metrics and four plots for token-position, early-prefix,
layer, and head aggregation. In that run, the first 1/2/4/8 tokens received
approximately 32.70% / 33.64% / 34.89% / 37.40% of measured attention mass;
token position 0 received the most. These are measured values for this prompt
and model, not a hard-coded universal conclusion. Requested prefix sizes longer
than the input are reported as unavailable.

---

# Approaches

## 1. Sliding Window

The sliding-window policy keeps the most recent tokens.

For a cache budget of `K`:

    Before eviction:

    [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9]

                        ↓ keep last K tokens

    After eviction:

    [t2, t3, t4, t5, t6, t7, t8, t9]

This is simple and provides a strong baseline.

### Advantages

- Very simple
- Predictable memory usage
- Low computational overhead
- No attention-score tracking required

### Limitation

Important information from older tokens can be discarded.

---

# 2. StreamingLLM-style Attention Sinks

StreamingLLM-style eviction preserves a small number of initial tokens, called **attention sinks**, while also keeping the most recent tokens.

For example:

    Before eviction:

    [s0, s1, t2, t3, t4, t5, t6, t7, t8, t9]

              ↓ preserve sinks + recent tokens

    After eviction:

    [s0, s1, t6, t7, t8, t9]

The initial sink tokens remain available even though they are no longer recent.

### Advantages

- Preserves initial context tokens
- Keeps recent context
- More structured than a pure sliding window

### Limitation

It does not dynamically identify which older tokens are actually important.

---

# 3. H2O-style Heavy Hitters

H2O uses attention scores to identify tokens that receive high accumulated attention.

The implementation:

1. Runs the model with attention outputs enabled.
2. Extracts attention for the newest query token.
3. Aggregates attention across attention heads.
4. Aggregates across layers.
5. Accumulates token-level attention scores.
6. Reserves configured sink tokens.
7. Reserves a configured recent local window.
8. Fills the remaining budget with the highest-scoring middle tokens.
9. Removes duplicates and restores original sequence order.

Conceptually:

    Tokens:

    t0  t1  t2  t3  t4  t5  t6  t7

    Scores:

    0.2 0.8 0.1 1.7 0.4 0.9 0.3 1.2

        ↓ sinks first, recent window second, then heavy hitters

    Keep:

    t0  t3  t5  t7

Selection is deterministic: sinks have first priority, recent tokens second,
and accumulated-score heavy hitters fill the remaining slots. Score ties are
broken by the lower cache index. The selected tokens are sorted back into
their original sequence order before constructing the new cache.

### Advantages

- Uses model attention to identify important tokens
- Can retain older tokens that remain relevant

### Limitation

- Requires attention-score computation
- Requires score tracking
- More expensive than purely positional policies

---

# Position Handling After Eviction

KV cache compression introduces an important positional issue.

Suppose the model originally processes:

    t0 t1 t2 t3 t4 t5 t6 t7

After eviction, the physical cache might contain:

    t0 t1 t6 t7

The physical cache length is now `4`, but `t6` and `t7` are still positions `6` and `7`.

They should not be renumbered as:

    t0 t1 t2 t3

Doing so can break the positional information expected by RoPE-based models.

Stage 6 therefore separates:

- **Physical cache position**
- **Absolute token position**

The generation code tracks the next absolute position independently from the current cache length.

With the installed Transformers 5 Qwen2 implementation, RoPE is computed
directly from `position_ids` and applied to keys before
`past_key_values.update`. Cached keys therefore already contain their original
rotation. Eviction slices those keys without recomputing or renumbering them,
and each new query receives its next absolute `position_ids` value. No manual
RoPE-table extension is required by this implementation.

Implemented in:

    src/position_utils.py

---

# Correctness Harness

Stage 7 introduces a deterministic, CPU-only correctness harness.

The purpose is to verify the cache manipulation logic before running model-quality or performance benchmarks.

`test_correctness_harness.py` contains deterministic synthetic checks.
`test_correctness.py` is the public entry point and separates:

- A: reference/cache agreement before eviction
- B: eviction indices, K/V correspondence, metadata, and budgets
- C: model-level Qwen middle-token position correctness
- D: descriptive compressed-vs-full generated-token agreement

Category D does not require exact equality after eviction. Model-loading
limitations are reported as `SKIP` rather than fabricated as passes.

## Sliding Window

Tests verify:

- Exact token selection
- No eviction below the cache budget
- Stable repeated application

## Attention Sink

Tests verify:

- Exact sink + recent-token selection
- No eviction below the cache budget
- Stable repeated application

## H2O Heavy Hitters

Tests verify:

- Exact heavy-hitter selection
- Original token ordering
- No eviction below the cache budget
- Batched attention-score support
- Attention-score/cache alignment
- Invalid score lengths
- Invalid score shapes

## Cache Managers

Tests verify:

- Correct manager behavior
- Cache updates
- Eviction statistics
- Repeated updates

## Position Handling

Tests verify:

- Absolute positions after eviction
- Correct `cache_position` generation

### Final Stage 7 commands

    python test_correctness_harness.py
    python test_correctness.py

---

# Project Structure

    attention-aware-kv-cache/
    │
    ├── results/
    │
    ├── src/
    │   ├── cache_utils.py
    │   ├── cache_manager.py
    │   ├── evictions.py
    │   ├── model_wrapper.py
    │   └── position_utils.py
    │
    ├── .gitignore
    ├── PROJECT_STATUS.md
    ├── README.md
    ├── requirements.txt
    │
    ├── stage4_experiment.py
    ├── test_correctness.py
    ├── test_attention_sink.py
    ├── test_correctness_harness.py
    ├── test_heavy_hitter.py
    ├── test_position_handling.py
    ├── test_sliding_window.py
    │
    └── visualise.py

---

# File Descriptions

## `src/model_wrapper.py`

Provides the model interface and generation implementations.

Contains generation paths for:

- Baseline generation
- Sliding-window caching
- StreamingLLM-style caching
- H2O heavy-hitter caching

It also handles:

- Absolute position IDs
- Absolute cache positions
- Attention-score extraction
- Attention-score accumulation
- Attention-mask handling
- Current/legacy Hugging Face cache inspection

---

## `src/cache_manager.py`

Contains cache-manager classes responsible for maintaining compressed KV caches.

Implemented managers:

    SlidingWindowCacheManager
    AttentionSinkCacheManager
    HeavyHitterCacheManager

The managers handle cache updates and track eviction statistics.

---

## `src/evictions.py`

Contains the core eviction utilities.

Implemented policies:

    sliding_window_evict()
    attention_sink_evict()
    heavy_hitter_evict()

The utilities operate on the cache representation used by the project and validate cache and score dimensions before performing eviction.

---

## `src/position_utils.py`

Contains utilities for position handling after cache eviction.

Implemented functionality includes:

- Absolute position ID generation
- Absolute cache-position generation
- Absolute positions independent of physical cache length

---

## `test_sliding_window.py`

Tests the sliding-window eviction implementation.

---

## `test_attention_sink.py`

Tests the attention-sink eviction implementation.

---

## `test_heavy_hitter.py`

Tests the H2O heavy-hitter implementation.

---

## `test_position_handling.py`

Tests position handling after KV cache eviction.

---

## `test_correctness_harness.py`

Stage 7 deterministic correctness suite.

It combines the core correctness checks for:

- Eviction policies
- Cache managers
- Attention-score alignment
- Input validation
- Position handling

---

## `visualise.py`

Runs the Stage 2 long-context attention measurement and writes plots plus JSON.

---

# Stages

| Stage | Description | Status |
|---|---|---|
| Stage 0 | Project foundation and scope | ✅ Complete |
| Stage 1 | Model loading and KV cache inspection | ✅ Complete |
| Stage 2 | Attention instrumentation and measured sink analysis | ✅ Complete |
| Stage 3 | Sliding-window KV cache eviction | ✅ Complete |
| Stage 4 | StreamingLLM-style attention-sink eviction | ✅ Complete |
| Stage 5 | H2O-style heavy-hitter eviction | ✅ Complete |
| Stage 6 | RoPE and position handling after eviction | ✅ Complete |
| Stage 7 | Full correctness harness | ✅ Complete |
| Stage 8 | Benchmarking and evaluation | ⏭️ Next |

---

# Stage 3 — Sliding Window

Implemented and validated the basic sliding-window cache policy.

The policy retains only the most recent tokens once the cache exceeds the configured budget.

Validation:

    python test_sliding_window.py

Expected:

    Stage 3 sliding-window tests passed.

---

# Stage 4 — StreamingLLM-style Attention Sinks

Implemented attention-sink-aware cache eviction.

The policy preserves a fixed number of initial sink tokens and the most recent tokens.

Validation:

    python test_attention_sink.py

Expected:

    Stage 4 attention-sink tests passed.

---

# Stage 5 — H2O Heavy Hitters

Implemented H2O-style heavy-hitter cache eviction based on accumulated attention scores.

Validation:

    python test_heavy_hitter.py

Expected:

    Stage 5 heavy-hitter tests passed.

The implementation was also integrated into generation.

At very small cache budgets, long generations can become repetitive. Such behavior is documented as an observation rather than a benchmark result.

---

# Stage 6 — RoPE and Position Handling

Stage 6 fixes positional handling after cache eviction.

The generation code now tracks:

    Absolute token position
            ≠
    Physical KV cache index

This allows compressed caches to retain tokens while preserving their original positional information.

Validation:

    python test_position_handling.py

Additional regression tests:

    python test_heavy_hitter.py
    python test_attention_sink.py
    python test_sliding_window.py

The Stage 6 position-handling and eviction tests pass.

---

# Stage 7 — Full Correctness Harness

Stage 7 adds a deterministic CPU-only correctness suite.

Run:

    python test_correctness_harness.py
    python test_correctness.py

The suites cover:

- Sliding-window exact selection
- Sliding-window no-op behavior
- Sliding-window repeated application
- Attention-sink exact selection
- Attention-sink no-op behavior
- Attention-sink repeated application
- H2O exact selection
- H2O original ordering
- H2O no-op behavior
- Batched attention scores
- Cache-manager behavior
- Score/cache alignment
- Invalid configurations
- Invalid attention-score lengths
- Invalid attention-score shapes
- Absolute positions after eviction
- Absolute cache positions
- Explicit original-position metadata
- H2O sink/recent/heavy-hitter reservations
- Reference-vs-custom cache agreement before eviction
- Model-level non-contiguous Qwen position handling
- Descriptive compressed-vs-full generated-token agreement

The quality comparison is descriptive and does not require compressed
generation to exactly equal full-cache generation.

---

# Installation

Clone the repository:

    git clone https://github.com/neelansh-gupta/attention-aware-kv-cache.git
    cd attention-aware-kv-cache

Create a virtual environment:

    python -m venv .venv

Activate it on Linux/macOS:

    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

---

# Running the Tests

Run the individual policy tests:

    python test_sliding_window.py
    python test_attention_sink.py
    python test_heavy_hitter.py
    python test_position_handling.py

Run the complete Stage 7 correctness harness:

    python test_correctness_harness.py
    python test_correctness.py

Run measured attention analysis:

    python visualise.py --max-tokens 256 --output-dir results/plots/attention

Run the real Stage 4 policy comparison:

    python stage4_experiment.py

---

# Current Validation Status

The current implementation has passed:

    Stage 3 sliding-window tests
    Stage 4 attention-sink tests
    Stage 5 heavy-hitter tests
    Stage 6 position-handling tests
    Stage 7 synthetic harness
    Stage 7 public A/B/C/D correctness suite

The model generation paths have also been exercised after the Stage 6 positional changes.

These generation runs confirm that the implementations execute successfully, but they are **not quantitative quality benchmarks**.

---

# Limitations

The current project focuses on implementation and correctness of KV cache compression.

It does not yet provide a complete quantitative comparison of:

- Generation latency
- Peak memory usage
- Throughput
- Perplexity
- Long-context benchmark performance
- Quality degradation at different cache budgets

These measurements are planned for the benchmarking stage.

---

# Next Stage

## Stage 8 — Benchmarking and Evaluation

The next stage will introduce quantitative experiments comparing:

- Baseline full KV cache
- Sliding Window
- StreamingLLM-style attention sinks
- H2O heavy hitters

Planned measurements include:

- Cache budget
- Compression ratio
- Generation latency
- Memory usage
- Generation behavior
- Comparative results across multiple cache budgets

The goal is to move from:

    Correct implementation
            ↓
    Correctness validation
            ↓
    Quantitative evaluation