from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch

from src.model_wrapper import ModelWrapper


MODEL_NAME = "Qwen/Qwen2.5-0.5B"

DEFAULT_PROMPT = (
    "Explain how transformer language models use attention and key-value "
    "caches during autoregressive text generation. Discuss why the KV cache "
    "grows with sequence length, why memory becomes a bottleneck for long "
    "contexts, and how different cache compression strategies can preserve "
    "important information while reducing memory usage. Compare simple "
    "recency-based approaches with attention-aware approaches and explain "
    "the trade-offs between memory, latency, and generation quality."
)

DEFAULT_BUDGETS = [32, 64, 128]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark KV-cache compression strategies."
    )

    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Hugging Face model name.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Number of new tokens generated per run.",
    )

    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=DEFAULT_BUDGETS,
        help="Cache budgets to evaluate.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repetitions per configuration.",
    )

    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Benchmark prompt.",
    )

    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for benchmark result files.",
    )

    return parser.parse_args()


def get_process_rss_mib() -> float:
    """
    Return the current process resident set size in MiB.

    On Linux, ru_maxrss is the process high-water RSS, not a resettable
    per-run peak. Therefore this metric is treated as contextual process
    memory rather than a direct comparison of KV-cache memory.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)

    # Linux reports ru_maxrss in KiB.
    if sys.platform.startswith("linux"):
        return usage.ru_maxrss / 1024.0

    # macOS reports ru_maxrss in bytes.
    return usage.ru_maxrss / (1024.0 * 1024.0)


def get_cuda_peak_memory_mib() -> float | None:
    """Return CUDA peak allocated memory if CUDA is available."""
    if not torch.cuda.is_available():
        return None

    return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)


def reset_cuda_peak_memory() -> None:
    """Reset CUDA's peak-memory counter."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def synchronize_device(device: torch.device) -> None:
    """Synchronize asynchronous accelerator operations when necessary."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except AttributeError:
            pass


def model_dtype_bytes(wrapper: ModelWrapper) -> int:
    """
    Return the number of bytes used by the model's parameter dtype.

    The KV-cache theoretical memory estimate assumes keys and values use
    the same dtype as the model parameters.
    """
    parameter = next(wrapper.model.parameters())
    return parameter.element_size()


def theoretical_kv_cache_bytes(
    wrapper: ModelWrapper,
    sequence_length: int,
) -> int:
    """
    Estimate KV-cache memory for a given sequence length.

    Formula:

        layers × 2(K,V) × sequence_length × KV_heads
        × head_dim × dtype_bytes

    For GQA models, num_key_value_heads is used when available.
    """
    config = wrapper.model.config

    num_layers = getattr(
        config,
        "num_hidden_layers",
        getattr(config, "n_layer", None),
    )

    num_kv_heads = getattr(
        config,
        "num_key_value_heads",
        getattr(config, "num_attention_heads", None),
    )

    hidden_size = getattr(config, "hidden_size", None)

    num_attention_heads = getattr(
        config,
        "num_attention_heads",
        None,
    )

    head_dim = getattr(config, "head_dim", None)

    if head_dim is None:
        if hidden_size is None or num_attention_heads is None:
            raise ValueError(
                "Unable to determine attention head dimension."
            )
        head_dim = hidden_size // num_attention_heads

    if num_layers is None or num_kv_heads is None:
        raise ValueError(
            "Unable to determine KV-cache dimensions from model config."
        )

    dtype_bytes = model_dtype_bytes(wrapper)

    return (
        num_layers
        * 2
        * sequence_length
        * num_kv_heads
        * head_dim
        * dtype_bytes
    )


def bytes_to_mib(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def compression_ratio(
    baseline_tokens: int,
    compressed_tokens: int,
) -> float:
    if compressed_tokens <= 0:
        return float("nan")

    return baseline_tokens / compressed_tokens


def memory_reduction_percent(
    baseline_bytes: int,
    compressed_bytes: int,
) -> float:
    if baseline_bytes <= 0:
        return float("nan")

    return 100.0 * (1.0 - compressed_bytes / baseline_bytes)


def distinct_2(text: str) -> float:
    """
    Calculate distinct-2, the fraction of unique token bigrams.

    This is a simple diversity diagnostic, not a quality benchmark.
    """
    tokens = text.split()

    if len(tokens) < 2:
        return 0.0

    bigrams = list(zip(tokens, tokens[1:]))

    return len(set(bigrams)) / len(bigrams)


def generated_token_count(
    wrapper: ModelWrapper,
    output_text: str,
    prompt_tokens: int,
) -> tuple[int, int]:
    """
    Determine generated and total token counts.

    ModelWrapper generation methods return the complete decoded sequence:
    prompt + generated tokens.

    Therefore:

        generated_tokens = total_tokens - prompt_tokens
    """
    output_ids = wrapper.tokenizer(
        output_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]

    total_tokens = int(output_ids.shape[-1])
    generated_tokens = max(0, total_tokens - prompt_tokens)

    return generated_tokens, total_tokens


def build_generation_function(
    wrapper: ModelWrapper,
    policy: str,
    budget: int | None,
    max_new_tokens: int,
) -> Callable[[str], str]:
    if policy == "baseline":
        return lambda prompt: wrapper.incremental_generate(
            prompt,
            max_new_tokens=max_new_tokens,
        )

    if policy == "sliding":
        if budget is None:
            raise ValueError("Sliding Window requires a budget.")

        return lambda prompt: wrapper.sliding_window_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            window_size=budget,
        )

    if policy == "streaming":
        if budget is None:
            raise ValueError("StreamingLLM requires a budget.")

        return lambda prompt: wrapper.streaming_llm_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            cache_budget=budget,
            sink_tokens=min(4, budget - 1),
        )

    if policy == "h2o":
        if budget is None:
            raise ValueError("H2O requires a budget.")

        return lambda prompt: wrapper.heavy_hitter_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            cache_budget=budget,
        )

    raise ValueError(f"Unknown policy: {policy}")


def benchmark_one_run(
    wrapper: ModelWrapper,
    prompt: str,
    policy: str,
    budget: int | None,
    max_new_tokens: int,
) -> dict:
    prompt_tokens = len(
        wrapper.tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]
    )

    generation_fn = build_generation_function(
        wrapper=wrapper,
        policy=policy,
        budget=budget,
        max_new_tokens=max_new_tokens,
    )

    reset_cuda_peak_memory()

    synchronize_device(wrapper.device)

    rss_before = get_process_rss_mib()

    start = time.perf_counter()

    output_text = generation_fn(prompt)

    synchronize_device(wrapper.device)

    elapsed = time.perf_counter() - start

    rss_after = get_process_rss_mib()
    cuda_peak = get_cuda_peak_memory_mib()

    generated_tokens, total_tokens = generated_token_count(
        wrapper,
        output_text,
        prompt_tokens,
    )

    # Because EOS is disabled during benchmarking, the expected count
    # should equal max_new_tokens. We keep the measured count above so
    # that the benchmark still detects unexpected behavior.
    expected_generated_tokens = max_new_tokens

    if generated_tokens != expected_generated_tokens:
        print(
            "WARNING: measured generated token count "
            f"{generated_tokens} != expected {expected_generated_tokens}"
        )

    if policy == "baseline":
        cache_tokens = total_tokens
    else:
        cache_tokens = int(budget)

    kv_bytes = theoretical_kv_cache_bytes(
        wrapper,
        cache_tokens,
    )

    return {
        "policy": policy,
        "budget": budget if budget is not None else "",
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_tokens": total_tokens,
        "cache_tokens": cache_tokens,
        "latency_seconds": elapsed,
        "tokens_per_second": (
            generated_tokens / elapsed
            if elapsed > 0
            else float("nan")
        ),
        "distinct_2": distinct_2(output_text),
        "kv_cache_bytes": kv_bytes,
        "kv_cache_mib": bytes_to_mib(kv_bytes),
        "rss_before_mib": rss_before,
        "rss_after_mib": rss_after,
        "cuda_peak_mib": (
            cuda_peak
            if cuda_peak is not None
            else ""
        ),
    }


def mean(values: list[float]) -> float:
    return statistics.mean(values)


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0

    return statistics.stdev(values)


def aggregate_results(
    raw_results: list[dict],
) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}

    for row in raw_results:
        key = (
            str(row["policy"]),
            str(row["budget"]),
        )
        groups.setdefault(key, []).append(row)

    summary: list[dict] = []

    baseline_group = groups.get(("baseline", ""), [])

    if not baseline_group:
        raise RuntimeError("Baseline benchmark results are missing.")

    baseline_total_tokens = int(
        round(
            mean(
                [
                    float(row["total_tokens"])
                    for row in baseline_group
                ]
            )
        )
    )

    baseline_kv_bytes = mean(
        [
            float(row["kv_cache_bytes"])
            for row in baseline_group
        ]
    )

    for (policy, budget), rows in groups.items():
        generated = [
            float(row["generated_tokens"])
            for row in rows
        ]

        total = [
            float(row["total_tokens"])
            for row in rows
        ]

        latency = [
            float(row["latency_seconds"])
            for row in rows
        ]

        throughput = [
            float(row["tokens_per_second"])
            for row in rows
        ]

        diversity = [
            float(row["distinct_2"])
            for row in rows
        ]

        kv_bytes = [
            float(row["kv_cache_bytes"])
            for row in rows
        ]

        rss_after = [
            float(row["rss_after_mib"])
            for row in rows
        ]

        cuda_values = [
            float(row["cuda_peak_mib"])
            for row in rows
            if row["cuda_peak_mib"] != ""
        ]

        representative_kv_bytes = mean(kv_bytes)

        summary.append(
            {
                "policy": policy,
                "budget": budget,
                "repeats": len(rows),
                "generated_tokens_mean": mean(generated),
                "generated_tokens_std": std(generated),
                "total_tokens_mean": mean(total),
                "total_tokens_std": std(total),
                "latency_seconds_mean": mean(latency),
                "latency_seconds_std": std(latency),
                "tokens_per_second_mean": mean(throughput),
                "tokens_per_second_std": std(throughput),
                "distinct_2_mean": mean(diversity),
                "distinct_2_std": std(diversity),
                "kv_cache_mib": bytes_to_mib(
                    representative_kv_bytes
                ),
                "compression_ratio": compression_ratio(
                    baseline_total_tokens,
                    int(
                        round(
                            mean(
                                [
                                    float(row["cache_tokens"])
                                    for row in rows
                                ]
                            )
                        )
                    ),
                ),
                "kv_memory_reduction_percent": memory_reduction_percent(
                    int(round(baseline_kv_bytes)),
                    int(round(representative_kv_bytes)),
                ),
                "rss_after_mib_mean": mean(rss_after),
                "rss_after_mib_std": std(rss_after),
                "cuda_peak_mib_mean": (
                    mean(cuda_values)
                    if cuda_values
                    else ""
                ),
                "cuda_peak_mib_std": (
                    std(cuda_values)
                    if cuda_values
                    else ""
                ),
            }
        )

    return summary


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    rows: list[dict],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            rows,
            file,
            indent=2,
        )


def print_summary(summary: list[dict]) -> None:
    print()
    print("=" * 125)
    print("BENCHMARK SUMMARY")
    print("=" * 125)

    header = (
        f"{'Policy':<12}"
        f"{'Budget':>8}"
        f"{'Gen Tok':>12}"
        f"{'Latency(s)':>18}"
        f"{'Tok/s':>18}"
        f"{'KV(MiB)':>12}"
        f"{'Compress':>12}"
        f"{'KV Reduct.':>14}"
    )

    print(header)
    print("-" * len(header))

    for row in summary:
        budget = row["budget"] if row["budget"] != "" else "-"

        generated = (
            f"{row['generated_tokens_mean']:.1f}"
            f"±{row['generated_tokens_std']:.1f}"
        )

        latency = (
            f"{row['latency_seconds_mean']:.3f}"
            f"±{row['latency_seconds_std']:.3f}"
        )

        throughput = (
            f"{row['tokens_per_second_mean']:.2f}"
            f"±{row['tokens_per_second_std']:.2f}"
        )

        print(
            f"{row['policy']:<12}"
            f"{str(budget):>8}"
            f"{generated:>12}"
            f"{latency:>18}"
            f"{throughput:>18}"
            f"{row['kv_cache_mib']:>12.3f}"
            f"{row['compression_ratio']:>12.2f}x"
            f"{row['kv_memory_reduction_percent']:>13.1f}%"
        )

    print("=" * 125)
    print()
    print(
        "Note: KV-cache memory is a theoretical calculation based on "
        "model configuration, sequence length, KV heads, head dimension, "
        "layers, and dtype."
    )
    print(
        "CPU RSS is a process high-water mark and should not be interpreted "
        "as per-policy KV-cache memory."
    )


def main() -> None:
    args = parse_args()

    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")

    if any(budget <= 0 for budget in args.budgets):
        raise ValueError("All budgets must be positive.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("Attention-Aware KV Cache Compression Benchmark")
    print("=" * 80)
    print(f"Model:           {args.model}")
    print(f"Max new tokens:  {args.max_new_tokens}")
    print(f"Budgets:         {args.budgets}")
    print(f"Repeats:         {args.repeats}")
    print()

    wrapper = ModelWrapper(
        model_name=args.model,
    )

    prompt_tokens = len(
        wrapper.tokenizer(
            args.prompt,
            add_special_tokens=False,
        )["input_ids"]
    )

    print(f"Prompt tokens:   {prompt_tokens}")
    print(f"Device:          {wrapper.device}")
    print()

    # Benchmark-only EOS suppression.
    #
    # This does NOT modify the model implementation. It ensures every
    # configuration produces exactly max_new_tokens so comparisons are
    # made at a fixed generation length.
    original_eos_token_id = wrapper.tokenizer.eos_token_id
    wrapper.tokenizer.eos_token_id = None

    raw_results: list[dict] = []

    configurations: list[tuple[str, int | None]] = [
        ("baseline", None),
    ]

    for policy in (
        "sliding",
        "streaming",
        "h2o",
    ):
        for budget in args.budgets:
            configurations.append(
                (policy, budget)
            )

    try:
        for policy, budget in configurations:
            label = (
                policy
                if budget is None
                else f"{policy} budget={budget}"
            )

            print(f"Running {label}...")

            for repeat in range(1, args.repeats + 1):
                result = benchmark_one_run(
                    wrapper=wrapper,
                    prompt=args.prompt,
                    policy=policy,
                    budget=budget,
                    max_new_tokens=args.max_new_tokens,
                )

                result["repeat"] = repeat

                raw_results.append(result)

                print(
                    f"  repeat {repeat}/{args.repeats}: "
                    f"{result['generated_tokens']} generated tokens, "
                    f"{result['latency_seconds']:.3f}s, "
                    f"{result['tokens_per_second']:.2f} tok/s"
                )

    finally:
        wrapper.tokenizer.eos_token_id = original_eos_token_id

    summary = aggregate_results(
        raw_results
    )

    raw_path = output_dir / "benchmark_raw.csv"
    summary_path = output_dir / "benchmark_summary.csv"
    json_path = output_dir / "benchmark_summary.json"

    write_csv(
        raw_path,
        raw_results,
    )

    write_csv(
        summary_path,
        summary,
    )

    write_json(
        json_path,
        summary,
    )

    print_summary(summary)

    print(f"Raw results:     {raw_path}")
    print(f"Summary CSV:     {summary_path}")
    print(f"Summary JSON:    {json_path}")


if __name__ == "__main__":
    main()