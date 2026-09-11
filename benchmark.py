"""Stage 8: reproducible KV-cache compression benchmark infrastructure.

This script measures wall-clock time and peak memory for existing generation
policies. It does not compute perplexity, Needle-in-a-Haystack scores, or
quality-vs-memory curves.

CLI:

    python benchmark.py --policy sliding --budget 128
    python benchmark.py --policy streaming --budget 128
    python benchmark.py --policy h2o --budget 128
    python benchmark.py --all --budget 128
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import torch
import transformers

from src.model_wrapper import GenerationResult, ModelWrapper


POLICIES = ("sliding", "streaming", "h2o")
DEFAULT_PROMPT = (
    "Explain how transformer language models use attention and key-value "
    "caches during autoregressive generation, and why dropping old tokens "
    "from the cache can change later predictions."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark KV-cache compression policies."
    )
    parser.add_argument(
        "--policy",
        choices=(*POLICIES, "full"),
        default=None,
        help="Single policy to run. Ignored when --all is set.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run sliding, streaming, and h2o under the same budget.",
    )
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--h2o-sink-tokens", type=int, default=1)
    parser.add_argument("--h2o-recent-window", type=int, default=1)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def process_rss_mib() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = float(usage.ru_maxrss)
    if sys.platform.startswith("darwin"):
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def environment_record(wrapper: ModelWrapper) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    cuda_name = None
    if cuda_available:
        cuda_name = torch.cuda.get_device_name(wrapper.device)
    return {
        "model": wrapper.model_name,
        "device": str(wrapper.device),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cuda_device_name": cuda_name,
    }


def generate(
    wrapper: ModelWrapper,
    policy: str,
    prompt: str,
    budget: int,
    max_new_tokens: int,
    sink_tokens: int,
    h2o_sink_tokens: int,
    h2o_recent_window: int,
) -> GenerationResult:
    if policy == "full":
        result = wrapper.incremental_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            return_diagnostics=True,
        )
    elif policy == "sliding":
        result = wrapper.sliding_window_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            window_size=budget,
            return_diagnostics=True,
        )
    elif policy == "streaming":
        result = wrapper.streaming_llm_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            cache_budget=budget,
            sink_tokens=min(sink_tokens, budget - 1),
            return_diagnostics=True,
        )
    elif policy == "h2o":
        result = wrapper.heavy_hitter_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            cache_budget=budget,
            sink_tokens=h2o_sink_tokens,
            recent_window=h2o_recent_window,
            return_diagnostics=True,
        )
    else:
        raise ValueError(f"Unknown policy: {policy}")
    if not isinstance(result, GenerationResult):
        raise TypeError("Generation did not return diagnostics.")
    return result


def run_one(
    wrapper: ModelWrapper,
    policy: str,
    prompt: str,
    budget: int,
    max_new_tokens: int,
    sink_tokens: int,
    h2o_sink_tokens: int,
    h2o_recent_window: int,
    repeat: int,
) -> dict[str, Any]:
    prompt_tokens = int(wrapper.tokenize(prompt)["input_ids"].shape[-1])
    if wrapper.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(wrapper.device)
        torch.cuda.synchronize(wrapper.device)
    rss_before = process_rss_mib()
    start = time.perf_counter()
    result = generate(
        wrapper,
        policy,
        prompt,
        budget,
        max_new_tokens,
        sink_tokens,
        h2o_sink_tokens,
        h2o_recent_window,
    )
    if wrapper.device.type == "cuda":
        torch.cuda.synchronize(wrapper.device)
    elapsed = time.perf_counter() - start
    rss_after = process_rss_mib()
    cuda_peak = None
    if wrapper.device.type == "cuda":
        cuda_peak = torch.cuda.max_memory_allocated(wrapper.device) / (1024.0 * 1024.0)
    generated = len(result.generated_token_ids)
    return {
        "policy": policy,
        "cache_budget": None if policy == "full" else budget,
        "repeat": repeat,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated,
        "context_length": prompt_tokens + generated,
        "cache_length": result.cache_length,
        "wall_clock_seconds": elapsed,
        "tokens_per_second": generated / elapsed if elapsed > 0 else None,
        "peak_memory_mib": cuda_peak if cuda_peak is not None else rss_after,
        "peak_memory_source": (
            "torch.cuda.max_memory_allocated"
            if cuda_peak is not None
            else "resource.ru_maxrss"
        ),
        "rss_before_mib": rss_before,
        "rss_after_mib": rss_after,
        "cuda_peak_allocated_mib": cuda_peak,
        "retained_positions": result.retained_positions,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [key for key in rows[0] if key != "retained_positions"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def main() -> None:
    args = parse_args()
    if args.budget <= 0:
        raise ValueError("--budget must be positive.")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    if args.all:
        policies = list(POLICIES)
    elif args.policy is not None:
        policies = [args.policy]
    else:
        raise ValueError("Specify --policy or --all.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wrapper = ModelWrapper(model_name=args.model or "Qwen/Qwen2.5-0.5B", device=args.device)
    environment = environment_record(wrapper)
    rows: list[dict[str, Any]] = []
    for policy in policies:
        print(f"Running policy={policy} budget={args.budget}")
        for repeat in range(1, args.repeats + 1):
            row = run_one(
                wrapper,
                policy,
                args.prompt,
                args.budget,
                args.max_new_tokens,
                args.sink_tokens,
                args.h2o_sink_tokens,
                args.h2o_recent_window,
                repeat,
            )
            rows.append(row)
            print(
                f"  repeat {repeat}: {row['generated_tokens']} tokens, "
                f"{row['wall_clock_seconds']:.3f}s, "
                f"cache_length={row['cache_length']}, "
                f"peak_memory_mib={row['peak_memory_mib']:.2f} "
                f"({row['peak_memory_source']})"
            )

    payload = {
        "stage": 8,
        "environment": environment,
        "settings": {
            "budget": args.budget,
            "max_new_tokens": args.max_new_tokens,
            "repeats": args.repeats,
            "policies": policies,
            "prompt": args.prompt,
        },
        "runs": rows,
    }
    json_path = output_dir / "benchmark_summary.json"
    csv_path = output_dir / "benchmark_raw.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(csv_path, rows)
    print(json.dumps({"environment": environment, "n_runs": len(rows)}, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
