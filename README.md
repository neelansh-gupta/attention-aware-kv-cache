# Attention-Aware KV Cache Compression

Postman AI/ML Recruitment Task 3 (Batch 25). Investigates why naively dropping
old tokens from a language model's KV cache destroys quality, and implements
and evaluates three eviction policies that try to do better under a fixed
cache budget:

1. **Sliding Window** — naive recency-only baseline (expected to fail).
2. **Attention-Sink-Aware (StreamingLLM-style)** — keep a small number of
   initial "sink" tokens plus a recent local window.
3. **Accumulated-Score / Heavy-Hitter (H2O-style)** — keep tokens that have
   historically received the most attention mass.

Model used: `Qwen/Qwen2.5-0.5B` (small enough to run on CPU or a free Colab
T4/laptop GPU).

## Project status

See [`PROJECT_STATUS.md`](./PROJECT_STATUS.md) for what is currently
implemented, what's next, and known issues. This project is built
incrementally in stages — do not assume anything beyond what
`PROJECT_STATUS.md` reports as complete.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10+. A CUDA GPU is not required — the code auto-selects
CPU/CUDA — but attention-instrumentation and benchmarking experiments will be
slow on CPU for longer contexts.

## Execution

Execution entry points are added stage by stage. Currently available:

- (none yet — Stage 0 is repository foundation only)

Planned, added in later stages:

```bash
python -m src.model_wrapper          # Stage 1: model + cache inspection
python -m src.model_wrapper --attn   # Stage 2: attention instrumentation
python test_correctness.py           # Stage 3+: correctness harness
python benchmark.py --policy sliding --budget 128   # Stage 8: benchmarking
python visualize.py                  # Stage 10: plots
```

## Roadmap

| Stage | Description | Status |
|---|---|---|
| 0 | Repository foundation | In progress |
| 1 | Model loading + KV cache inspection | Not started |
| 2 | Attention instrumentation + sink analysis | Not started |
| 3 | Sliding window eviction | Not started |
| 4 | StreamingLLM / attention-sink policy | Not started |
| 5 | H2O / heavy-hitter policy | Not started |
| 6 | Correct RoPE / position handling after eviction | Not started |
| 7 | Full correctness harness | Not started |
| 8 | Benchmark infrastructure | Not started |
| 9 | Perplexity + Needle-in-a-Haystack evaluation | Not started |
| 10 | Visualization + quality-vs-memory curves | Not started |
| 11 | WRITEUP.md | Not started |
| 12 | Final repository audit | Not started |

## Repository structure

```text
ai-ml/
├── src/
│   ├── __init__.py
│   ├── model_wrapper.py      (Stage 1+)
│   ├── cache_manager.py      (Stage 3+)
│   └── evictions.py          (Stage 3+)
├── test_correctness.py       (Stage 3+, expanded Stage 7)
├── benchmark.py               (Stage 8)
├── visualize.py                (Stage 2+, expanded Stage 10)
├── requirements.txt
├── README.md
├── WRITEUP.md                  (Stage 11)
├── PROJECT_STATUS.md
└── results/
    └── plots/
```

## Notes on scope

Per the task brief, incomplete submissions are expected. This repo reports
honestly in `PROJECT_STATUS.md` and `WRITEUP.md` which experiments have
actually been run — no fabricated perplexity, NIH accuracy, memory, or
benchmark numbers.
