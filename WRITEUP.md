# Implementation Note: RoPE and Cache Positions

This is the Stage 6 correctness note, not the final Stage 11 experimental
writeup.

## Mechanism used by the installed Qwen2 implementation

The verified environment uses Transformers 5.16.1 with
`Qwen/Qwen2.5-0.5B`. Inspection of `Qwen2Attention.forward` shows this order:

1. project the current hidden states into queries, keys, and values;
2. compute rotary cosine/sine values from the supplied `position_ids`;
3. apply RoPE to the current queries and keys;
4. append the already-rotated keys to `past_key_values`.

Consequently, a cached key retains the rotary encoding of its original
position. Evicting a middle token must select the existing cached K/V entries
without recomputing or renumbering them. For each subsequent token, the
generation loop supplies the next absolute `position_ids` value independently
of the shortened physical cache length.

For example, after retaining original positions `[0, 1, 2, 7, 8, 9]`, the
next query uses position `10`; it must not use physical-cache position `6`.
No manual RoPE cache-extension helper is used because this Qwen2 rotary module
computes embeddings directly from `position_ids`.

## Verification boundary

`test_position_handling.py` runs the real model, verifies that non-contiguous
eviction retains K/V tensors exactly from original positions
`[0, 1, 2, 7, 8, 9]`, and captures the next rotary call to confirm position
`10` rather than `6`. This establishes position-handling correctness for the
installed Qwen2/Transformers implementation. It does not claim compatibility
with every historical or third-party RoPE implementation.
