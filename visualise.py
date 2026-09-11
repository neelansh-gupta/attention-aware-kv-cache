"""Stage 2 attention instrumentation and plotting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.model_wrapper import ModelWrapper


DEFAULT_TEXT = (
    "Attention mechanisms let language models retrieve information from prior "
    "tokens. A long-running assistant must preserve instructions, facts, and "
    "recent dialogue while its key-value cache grows. "
) * 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure and plot Qwen attention.")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output-dir", default="results/plots/attention")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_tokens < 8:
        raise ValueError("--max-tokens must be at least 8")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wrapper = ModelWrapper(device=args.device)
    encoded = wrapper.tokenize(args.text)["input_ids"][0, : args.max_tokens]
    measured_text = wrapper.tokenizer.decode(encoded, skip_special_tokens=False)
    attentions, token_info = wrapper.inspect_attention(measured_text)
    report = wrapper.analyze_attention_sinks(attentions)
    analysis = report["analysis"]

    received = analysis.attention_received.numpy()
    positions = np.arange(report["sequence_length"])
    layer_received = np.stack(
        [layer.mean(dim=(0, 1)).numpy() for layer in analysis.attention_by_layer]
    )
    head_received = analysis.attention_by_head.mean(dim=1).numpy()

    plt.figure(figsize=(11, 4))
    plt.plot(positions, received)
    plt.xlabel("Token position")
    plt.ylabel("Attention mass received (sum over queries)")
    plt.title("Aggregate attention received by token position")
    plt.tight_layout()
    plt.savefig(output_dir / "attention_by_token_position.png", dpi=150)
    plt.close()

    available = {
        count: value
        for count, value in report["early_token_attention"].items()
        if value is not None
    }
    plt.figure(figsize=(7, 4))
    plt.plot(list(available), list(available.values()), marker="o")
    plt.xticks(list(available))
    plt.xlabel("Number of earliest tokens")
    plt.ylabel("Fraction of total measured attention")
    plt.title("Attention received by early-token prefixes")
    plt.tight_layout()
    plt.savefig(output_dir / "early_token_attention.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.imshow(layer_received, aspect="auto", interpolation="nearest")
    plt.xlabel("Token position")
    plt.ylabel("Transformer layer")
    plt.title("Attention received across layers")
    plt.colorbar(label="Mean attention over heads and queries")
    plt.tight_layout()
    plt.savefig(output_dir / "layer_attention_heatmap.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.imshow(head_received, aspect="auto", interpolation="nearest")
    plt.xlabel("Token position")
    plt.ylabel("Attention head (averaged across layers)")
    plt.title("Attention received across heads")
    plt.colorbar(label="Mean attention over layers and queries")
    plt.tight_layout()
    plt.savefig(output_dir / "head_attention_heatmap.png", dpi=150)
    plt.close()

    early = report["early_token_attention"]
    uniform_first_8 = 8 / report["sequence_length"]
    measured_first_8 = early[8]
    observation = (
        "First-8 attention exceeds a uniform-position baseline."
        if measured_first_8 is not None and measured_first_8 > uniform_first_8
        else "First-8 attention does not exceed a uniform-position baseline."
    )
    metrics = {
        "model": wrapper.model_name,
        "device": str(wrapper.device),
        "sequence_length": report["sequence_length"],
        "layers": len(analysis.attention_by_layer),
        "heads": int(analysis.attention_by_head.shape[0]),
        "most_attended_token_position": report["most_attended_token_position"],
        "early_token_attention_fraction": {
            str(count): value for count, value in early.items()
        },
        "uniform_first_8_fraction": uniform_first_8,
        "observation": observation,
        "plot_files": sorted(path.name for path in output_dir.glob("*.png")),
        "token_count": len(token_info),
    }
    (output_dir / "attention_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    print(json.dumps(metrics, indent=2))
    print(f"Plots written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
