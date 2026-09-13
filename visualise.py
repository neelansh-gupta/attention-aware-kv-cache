"""Stage 2/10 plotting from saved experiment results.

Default mode reads JSON/CSV/NPZ artifacts and writes plots. It does not
hard-code measured values. Optional --collect-attention re-runs the model
to refresh attention arrays.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TEXT = (
    "Attention mechanisms let language models retrieve information from prior "
    "tokens. A long-running assistant must preserve instructions, facts, and "
    "recent dialogue while its key-value cache grows. "
) * 100

RESULTS = Path("results")
ATTENTION_DIR = RESULTS / "plots" / "attention"
QUALITY_DIR = RESULTS / "plots" / "quality"
MEMORY_DIR = RESULTS / "plots" / "memory"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot attention and quality-vs-memory from saved results."
    )
    parser.add_argument(
        "--collect-attention",
        action="store_true",
        help="Re-run attention collection and save arrays plus plots.",
    )
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def collect_attention(args: argparse.Namespace, attention_dir: Path) -> None:
    from src.model_wrapper import ModelWrapper

    if args.max_tokens < 8:
        raise ValueError("--max-tokens must be at least 8")
    attention_dir.mkdir(parents=True, exist_ok=True)
    wrapper = ModelWrapper(device=args.device)
    encoded = wrapper.tokenize(args.text)["input_ids"][0, : args.max_tokens]
    measured_text = wrapper.tokenizer.decode(encoded, skip_special_tokens=False)
    attentions, token_info = wrapper.inspect_attention(measured_text)
    report = wrapper.analyze_attention_sinks(attentions)
    analysis = report["analysis"]
    received = analysis.attention_received.numpy()
    layer_received = np.stack(
        [layer.mean(dim=(0, 1)).numpy() for layer in analysis.attention_by_layer]
    )
    head_received = analysis.attention_by_head.mean(dim=1).numpy()
    np.savez(
        attention_dir / "attention_arrays.npz",
        received=received,
        layer_received=layer_received,
        head_received=head_received,
    )
    early = {
        str(count): value for count, value in report["early_token_attention"].items()
    }
    metrics = {
        "model": wrapper.model_name,
        "device": str(wrapper.device),
        "sequence_length": report["sequence_length"],
        "layers": len(analysis.attention_by_layer),
        "heads": int(analysis.attention_by_head.shape[0]),
        "most_attended_token_position": report["most_attended_token_position"],
        "early_token_attention_fraction": early,
        "uniform_first_8_fraction": 8 / report["sequence_length"],
        "observation": (
            "First-8 attention exceeds a uniform-position baseline."
            if early.get("8") is not None
            and early["8"] > 8 / report["sequence_length"]
            else "First-8 attention does not exceed a uniform-position baseline."
        ),
        "token_count": len(token_info),
        "arrays_file": "attention_arrays.npz",
    }
    (attention_dir / "attention_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )


def plot_attention(attention_dir: Path) -> list[str]:
    written: list[str] = []
    metrics_path = attention_dir / "attention_metrics.json"
    arrays_path = attention_dir / "attention_arrays.npz"
    if not metrics_path.exists() and not arrays_path.exists():
        print("SKIP: no saved attention metrics or arrays.")
        return written

    if arrays_path.exists():
        arrays = np.load(arrays_path)
        received = arrays["received"]
        positions = np.arange(len(received))
        plt.figure(figsize=(11, 4))
        plt.plot(positions, received)
        plt.xlabel("Token position")
        plt.ylabel("Attention mass received (sum over queries)")
        plt.title("Aggregate attention received by token position")
        plt.tight_layout()
        plt.savefig(attention_dir / "attention_by_token_position.png", dpi=150)
        plt.close()
        written.append("attention_by_token_position.png")

        plt.figure(figsize=(11, 6))
        plt.imshow(arrays["layer_received"], aspect="auto", interpolation="nearest")
        plt.xlabel("Token position")
        plt.ylabel("Transformer layer")
        plt.title("Attention received across layers")
        plt.colorbar(label="Mean attention over heads and queries")
        plt.tight_layout()
        plt.savefig(attention_dir / "layer_attention_heatmap.png", dpi=150)
        plt.close()
        written.append("layer_attention_heatmap.png")

        plt.figure(figsize=(11, 5))
        plt.imshow(arrays["head_received"], aspect="auto", interpolation="nearest")
        plt.xlabel("Token position")
        plt.ylabel("Attention head (averaged across layers)")
        plt.title("Attention received across heads")
        plt.colorbar(label="Mean attention over layers and queries")
        plt.tight_layout()
        plt.savefig(attention_dir / "head_attention_heatmap.png", dpi=150)
        plt.close()
        written.append("head_attention_heatmap.png")
    else:
        print(
            "Note: attention heatmaps were not regenerated because "
            f"{arrays_path} is missing. Existing PNG files were left unchanged."
        )

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        early = {
            int(count): value
            for count, value in metrics["early_token_attention_fraction"].items()
            if value is not None
        }
        plt.figure(figsize=(7, 4))
        plt.plot(list(early), list(early.values()), marker="o")
        plt.xticks(list(early))
        plt.xlabel("Number of earliest tokens")
        plt.ylabel("Fraction of total measured attention")
        plt.title("Attention received by early-token prefixes")
        plt.tight_layout()
        plt.savefig(attention_dir / "early_token_attention.png", dpi=150)
        plt.close()
        written.append("early_token_attention.png")
    return written


def load_perplexity_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    paths = list(results_dir.glob("quality/budget_*/stage9_perplexity.json"))
    fallback = results_dir / "stage9_perplexity.json"
    if fallback.exists():
        paths.append(fallback)
    seen = set()
    for path in paths:
        payload = json.loads(path.read_text())
        for row in payload.get("results", []):
            budget = row.get("cache_budget")
            if row["policy"] == "full":
                budget = payload.get("document_tokens", "full")
            key = (row["policy"], budget, round(float(row["perplexity"]), 6))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "policy": row["policy"],
                    "cache_budget": row.get("cache_budget"),
                    "document_tokens": payload.get("document_tokens"),
                    "perplexity": row["perplexity"],
                    "mean_nll": row["mean_nll"],
                    "source": str(path),
                }
            )
    return rows


def load_nih_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    paths = list(results_dir.glob("quality/budget_*/stage9_nih.json"))
    fallback = results_dir / "stage9_nih.json"
    if fallback.exists():
        paths.append(fallback)
    seen = set()
    for path in paths:
        payload = json.loads(path.read_text())
        budget = payload.get("cache_budget")
        grouped: dict[str, list[bool]] = {}
        for row in payload.get("results", []):
            grouped.setdefault(row["policy"], []).append(bool(row["success"]))
        for policy, outcomes in grouped.items():
            policy_budget = None if policy == "full" else budget
            key = (policy, policy_budget)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "policy": policy,
                    "cache_budget": policy_budget,
                    "accuracy": sum(outcomes) / len(outcomes),
                    "n_cases": len(outcomes),
                    "source": str(path),
                }
            )
    return rows


def load_benchmark_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple] = set()
    paths = [results_dir / "benchmark_raw.csv"]
    paths.extend(sorted(results_dir.glob("benchmark/budget_*/benchmark_raw.csv")))
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                budget = int(row["cache_budget"]) if row["cache_budget"] else None
                key = (row["policy"], budget)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "policy": row["policy"],
                        "cache_budget": budget,
                        "peak_memory_mib": float(row["peak_memory_mib"]),
                        "peak_memory_source": row["peak_memory_source"],
                        "wall_clock_seconds": float(row["wall_clock_seconds"]),
                        "source": str(path),
                    }
                )
    return rows


def plot_series(
    rows: list[dict],
    y_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
    skip_full: bool = True,
) -> None:
    by_policy: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        if skip_full and row["policy"] == "full":
            continue
        if row["cache_budget"] is None:
            continue
        by_policy.setdefault(row["policy"], []).append(
            (float(row["cache_budget"]), float(row[y_key]))
        )
    if not by_policy:
        print(f"SKIP: no plotted points for {output_path.name}")
        return
    plt.figure(figsize=(8, 5))
    for policy, points in sorted(by_policy.items()):
        points = sorted(points)
        plt.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            label=policy,
        )
    plt.xlabel("Cache budget (tokens)")
    plt.ylabel(ylabel)
    plt.title(title)
    full_values = [float(row[y_key]) for row in rows if row["policy"] == "full"]
    if full_values:
        plt.axhline(
            sum(full_values) / len(full_values),
            linestyle="--",
            label="full cache",
        )
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def theoretical_kv_mib(budget: int) -> float:
    """KV size from Qwen2.5-0.5B config: 24 layers, 2 KV heads, 64 dim, 2-byte dtype."""
    return (24 * 2 * budget * 2 * 64 * 2) / (1024.0 * 1024.0)


def plot_theoretical_memory(budgets: list[int], output_path: Path) -> None:
    if not budgets:
        print(f"SKIP: no budgets for {output_path.name}")
        return
    budgets = sorted(set(budgets))
    plt.figure(figsize=(8, 5))
    plt.plot(budgets, [theoretical_kv_mib(budget) for budget in budgets], marker="o")
    plt.xlabel("Cache budget (tokens)")
    plt.ylabel("Theoretical KV cache (MiB)")
    plt.title("Theoretical KV memory vs cache budget")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    attention_dir = results_dir / "plots" / "attention"
    quality_dir = results_dir / "plots" / "quality"
    memory_dir = results_dir / "plots" / "memory"
    if args.collect_attention:
        collect_attention(args, attention_dir)

    written = plot_attention(attention_dir)
    ppl_rows = load_perplexity_rows(results_dir)
    nih_rows = load_nih_rows(results_dir)
    bench_rows = load_benchmark_rows(results_dir)

    plot_series(
        ppl_rows,
        "perplexity",
        "Perplexity",
        "Perplexity vs cache budget",
        quality_dir / "perplexity_vs_budget.png",
    )
    plot_series(
        nih_rows,
        "accuracy",
        "NIH success rate (fraction of depths)",
        "Needle-in-a-Haystack accuracy vs cache budget",
        quality_dir / "nih_accuracy_vs_budget.png",
    )
    plot_series(
        bench_rows,
        "peak_memory_mib",
        "Peak memory (MiB)",
        "Measured peak memory vs cache budget",
        memory_dir / "peak_memory_vs_budget.png",
        skip_full=False,
    )
    budgets = [
        int(row["cache_budget"])
        for row in ppl_rows + nih_rows + bench_rows
        if row.get("cache_budget") is not None
    ]
    plot_theoretical_memory(budgets, memory_dir / "theoretical_kv_vs_budget.png")

    index = {
        "attention_plots": written,
        "perplexity_points": ppl_rows,
        "nih_points": nih_rows,
        "benchmark_points": bench_rows,
        "sources_are_saved_results": True,
    }
    (results_dir / "plots" / "stage10_plot_index.json").write_text(
        json.dumps(index, indent=2) + "\n"
    )
    print(json.dumps({"attention_plots": written, "ppl_n": len(ppl_rows), "nih_n": len(nih_rows), "bench_n": len(bench_rows)}, indent=2))
    print(f"Wrote plots under {results_dir / 'plots'}")


if __name__ == "__main__":
    main()
