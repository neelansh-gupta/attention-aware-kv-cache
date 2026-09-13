# Attention-Aware KV Cache Compression

Postman AI/ML Recruitment Task 3. This writeup reports only measurements
that exist in `results/`. It does not substitute for a large-scale study.

## 1. Motivation

Autoregressive transformers store a key and value vector for every past
token. That KV cache grows linearly with context, so a long-running agent
eventually has to drop entries. Dropping the oldest tokens is the obvious
fix, but it can destroy the information the model still attends to. This
project instruments attention on `Qwen/Qwen2.5-0.5B`, implements three
eviction policies, preserves original RoPE identities after middle-token
eviction, and compares the policies under a shared evaluation protocol.

## 2. Attention sinks

On a measured 256-token CPU prompt, token position 0 received the most
attention. The first 1/2/4/8 tokens received 32.70% / 33.64% / 34.89% /
37.40% of aggregated attention mass. A uniform-position baseline for the
first eight tokens would be 8/256 = 3.125%. The first eight tokens therefore
absorbed far more mass than their share of the sequence.

That pattern is the operational meaning of an attention sink: early tokens,
especially the first, collect a disproportionate fraction of later queries'
attention even when they are not uniquely informative. Softmax over a long
key sequence needs somewhere to put residual mass; initial tokens are a
stable place to put it. The observation is prompt-specific. It is not a
claim that every sequence or every layer behaves identically. Layer and
head heatmaps are in `results/plots/attention/`.

## 3. Sliding Window

Sliding Window keeps the most recent `K` tokens and discards everything
older. It is cheap and has a hard memory bound. It cannot keep an early
system instruction or a fact that is no longer in the local window.

In the Stage 4 real-model comparison (34-token prompt, budget 8), Sliding
Window retained `[29, 30, 31, 32, 33, 34, 35, 36]`. That is exactly the
newest eight positions.

## 4. StreamingLLM

StreamingLLM-style eviction keeps a configured number of initial sink
tokens plus a recent local window, with no duplicate indices, restoring
sequence order. The Stage 4 run with two sinks retained
`[0, 1, 31, 32, 33, 34, 35, 36]`. The first two tokens survived even though
they were no longer recent. That is the structural reason StreamingLLM can
preserve a short system prefix that Sliding Window deletes.

## 5. H2O

H2O in this repository is not pure top-k. Selection is deterministic:

1. reserve `sink_tokens` (default 1);
2. reserve a recent local window;
3. fill remaining slots with the highest accumulated attention scores,
   breaking ties by lower original index;
4. return indices in original sequence order.

Scores are the newest query's attention, averaged over heads then layers,
accumulated across decode or teacher-forced steps. Tokens that remain in
cache keep adding score; tokens that have been present longer have more
opportunities to accumulate. That is the early-token bias discussed below.

## 6. RoPE position handling

Installed Transformers 5.16.1 Qwen2 applies RoPE to the current keys
*before* `past_key_values.update`. Cached keys already carry their original
rotations. Eviction must slice those entries without renumbering them. New
tokens receive the next absolute `position_ids`.

The model-level regression uses original positions `0 1 2 3 4 5 6 7 8 9`
and retains `[0, 1, 2, 7, 8, 9]`. Retained K/V match the full cache at those
indices. The next rotary call receives position `10`, not shortened-cache
position `6`. A helper that extended a cosine/sine table was unnecessary
for this Qwen2 implementation and was removed.

## 7. Experimental setup

Environment: Python 3.14.4, PyTorch 2.14.0+cu130, Transformers 5.16.1,
`Qwen/Qwen2.5-0.5B` on CPU, `cuda_available: false`.

Perplexity and NIH walk the same tokens one at a time. Compressed policies
evict after every token. `full` never evicts. Perplexity is
`exp(mean next-token NLL)` on a 64-token document. NIH inserts the unique
passcode `ZX9QWERTY7731` at start, middle, and end of a filler haystack
(~80 prompt tokens) and asks for it. Success requires the full passcode in
a 24-token greedy continuation. Quality budgets measured: 16 and 32.
Streaming used 4 sink tokens. H2O used 1 sink token and a recent window of
8. Plots in `results/plots/` are generated from those saved files.

An 8-token NIH attempt truncated even the full-cache answer and is not
used. Wall-clock times from Stage 8 were collected in separate process
runs and are not treated as a controlled latency ranking.

## 8. Perplexity results

| Policy    | Budget 16 | Budget 32 |
|-----------|-----------|-----------|
| full      | 49.75     | 49.75     |
| sliding   | 379.16    | 225.09    |
| streaming | 118.32    | 103.07    |
| h2o       | 104.98    | 52.00     |

Full-cache perplexity is independent of budget because that policy does not
evict. Sliding Window is worst at both budgets. StreamingLLM is better than
sliding but worse than H2O. At budget 32, H2O (52.00) is close to full
(49.75). Increasing the budget from 16 to 32 reduced sliding and H2O
perplexity substantially; streaming improved only modestly.

## 9. NIH results

At both budgets, full cache recovered the passcode at start, middle, and
end (accuracy 1.0). Sliding, StreamingLLM, and H2O recovered it at none of
the three depths (accuracy 0.0).

Retention of needle tokens still differed. At budget 32, sliding kept 0/21
needle tokens at every depth. Streaming kept 4/21 at start (the sink
prefix) and 0 at middle and end. H2O kept 13/21, 8/21, and 3/21 at
start/middle/end, then still emitted wrong codes (`ZX90`, `ZXZ`). At budget
16, H2O's retained needle fraction fell further (8/21, 2/21, 0/21).

This is the intended diagnostic: perplexity can look acceptable while the
fact is gone. H2O at budget 32 nearly matched full perplexity and still
failed retrieval.

## 10. Memory results

Theoretical KV size for this model config (24 layers, 2 KV heads, head dim
64, 2-byte dtype) is `24 × 2 × budget × 2 × 64 × 2` bytes, i.e. 0.188 /
0.375 / 0.750 MiB at budgets 16 / 32 / 64. That quantity scales with
budget by construction.

Measured peak RSS was 1761.86–1762.14 MiB across those budgets and all
three policies (`resource.ru_maxrss`). On CPU that high-water mark is
dominated by model weights, not by the KV tensors, so it does not isolate
cache compression. CUDA peak allocation was not measured.

## 11. Quality-vs-memory analysis

Against theoretical KV size, H2O gave the best perplexity per retained
token in this setting, then StreamingLLM, then Sliding Window. NIH accuracy
did not improve when the budget doubled from 16 to 32 for any compressed
policy. Process RSS was effectively constant, so a quality-vs-RSS curve is
not informative here. A quality-vs-theoretical-KV curve is: H2O at 32
tokens (~0.375 MiB of KV) almost matches full-document perplexity, while
retrieval remains failed. More budgets, longer documents, and GPU memory
traces were not collected.

## 12. H2O early-token bias

Accumulated attention is a running sum. A token that is cached at step 1
can receive score at every later step; a token that appears late cannot.
Even with recency reservation, leftover slots therefore tend to go to
tokens that have been visible longer. The PPL walk at budget 32 retained
H2O positions including `0–13` plus a recent tail, whereas Sliding Window
retained only `31–62`. NIH start retention (13/21 vs streaming's 4/21)
shows the same early bias. Early bias can help fluent local prediction and
still fail to keep a specific 21-token fact intact.

## 13. Recommendation for multi-turn agents

A multi-turn agent needs a stable prefix (instructions, tools, identity)
and a recent dialogue window. StreamingLLM is the policy that encodes that
structure directly. Sliding Window will drop the prefix as soon as the
conversation exceeds `K`. H2O may keep some early tokens, but which ones
depends on accumulated scores, not on a guaranteed instruction span.

This project's NIH start case did not show StreamingLLM actually retrieving
the passcode: only four sink tokens of a 21-token needle survived. The
recommendation is therefore structural, not a claim of measured retrieval
success. For agents, use StreamingLLM with a sink long enough to cover the
system prompt, and treat H2O as optional extra capacity only if score
accumulation is trusted.

## 14. Recommendation for long-document summarizers

Summarization needs scattered content, not only a prefix and a tail.
Sliding Window discards the beginning of the document. StreamingLLM keeps
only a tiny prefix beyond the tail. H2O is the policy meant to keep
high-attention middle tokens, and it produced the only compressed
perplexity close to full cache.

It still failed NIH at budgets 16 and 32, so it is not sufficient by
itself at these sizes. For summarization, H2O (or a larger budget, or
per-layer budgets, which were not implemented) is the better starting
point among the three. Do not use Sliding Window if early sections matter.
Do not use NIH-style exact-fact recovery as a proxy for summary quality;
that task was not measured.

## 15. Limitations

- CPU only; no CUDA memory or latency study.
- Attention analysis used one 256-token prompt.
- Quality evaluation used a 64-token document and two budgets.
- NIH used one passcode, three depths, and 24 generated tokens.
- Compressed NIH accuracy was zero; larger budgets were not tested.
- Peak RSS does not measure KV-cache bytes.
- Stretch per-head / per-layer budgets were not implemented.
- Results should not be treated as a ranking of published StreamingLLM or
  H2O papers. They describe this implementation, model, and protocol.
