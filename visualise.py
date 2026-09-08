from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.model_wrapper import ModelWrapper


DEFAULT_TEXT = """
Artificial intelligence has transformed the way computers process
language, reason about information, and interact with humans. Modern
language models use attention mechanisms to determine which tokens
are important when producing each new prediction. As context becomes
longer, understanding how attention is distributed across token
positions becomes increasingly important for efficient inference.
""" * 8


def save_attention_distribution(
    analysis,
    output_path: Path,
):
    """
    Plot average attention received by each token position.
    """

    received = (
        analysis.token_attention_received.numpy()
    )

    positions = np.arange(
        len(received)
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        positions,
        received,
    )

    plt.xlabel("Token position")
    plt.ylabel("Average attention received")
    plt.title("Attention Received by Token Position")

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def save_early_token_attention(
    analysis,
    output_path: Path,
):
    """
    Plot cumulative attention received by the first
    1, 2, 4 and 8 tokens.
    """

    token_counts = [1, 2, 4, 8]

    values = analysis.early_token_attention(
        token_counts=token_counts
    )

    x = list(values.keys())
    y = list(values.values())

    plt.figure(figsize=(8, 5))

    plt.plot(
        x,
        y,
        marker="o",
    )

    plt.xlabel("Number of earliest tokens")
    plt.ylabel("Fraction of total attention received")
    plt.title("Attention Concentration on Early Tokens")

    plt.xticks(token_counts)

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def save_layer_attention_heatmap(
    analysis,
    output_path: Path,
):
    """
    Create a heatmap showing attention received by token position
    separately for every layer.

    Each row corresponds to one transformer layer.
    """

    layer_vectors = []

    for layer_attention in (
        analysis.attention_by_layer
    ):
        # [heads, query, key]
        mean_over_heads = (
            layer_attention.mean(dim=0)
        )

        # Average over query positions.
        received = mean_over_heads.mean(dim=0)

        layer_vectors.append(
            received.numpy()
        )

    matrix = np.stack(
        layer_vectors,
        axis=0,
    )

    plt.figure(
        figsize=(14, 7)
    )

    plt.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
    )

    plt.xlabel("Token position")
    plt.ylabel("Transformer layer")
    plt.title(
        "Attention Received by Token Position Across Layers"
    )

    plt.colorbar(
        label="Average attention received"
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def save_head_attention_heatmap(
    analysis,
    output_path: Path,
):
    """
    Visualize attention received by token position for every
    attention head in the first transformer layer.

    This provides a direct view of whether concentration on early
    tokens is consistent across heads or isolated to specific heads.
    """

    first_layer = (
        analysis.attention_by_layer[0]
    )

    # Shape:
    # [heads, query_length, key_length]

    received_per_head = (
        first_layer.mean(dim=1)
    )

    matrix = received_per_head.numpy()

    plt.figure(
        figsize=(14, 7)
    )

    plt.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
    )

    plt.xlabel("Token position")
    plt.ylabel("Attention head")
    plt.title(
        "First-Layer Attention Received by Token Position Across Heads"
    )

    plt.colorbar(
        label="Average attention received"
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def print_analysis_report(
    result,
):
    print("\nAttention Analysis")
    print("==================")

    print(
        f"Sequence length: "
        f"{result['sequence_length']}"
    )

    print(
        f"Most attended token position: "
        f"{result['most_attended_token_position']}"
    )

    print(
        "\nAttention received by earliest tokens:"
    )

    for count, value in (
        result["early_token_attention"].items()
    ):
        print(
            f"First {count:>2} token(s): "
            f"{value:.6f} "
            f"({value * 100:.2f}%)"
        )

    print(
        "\nNote:"
    )
    print(
        "These measurements describe attention concentration. "
        "They do not by themselves establish that an attention "
        "sink exists."
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze attention concentration and "
            "early-token attention."
        )
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to analyze.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/attention",
        help="Directory for generated plots.",
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    text = (
        args.text
        if args.text is not None
        else DEFAULT_TEXT
    )

    print(
        "Loading model..."
    )

    wrapper = ModelWrapper()

    print(
        "Running attention analysis..."
    )

    result = (
        wrapper.analyze_attention_sinks(
            text
        )
    )

    analysis = result["analysis"]

    print_analysis_report(
        result
    )

    # --------------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------------

    save_attention_distribution(
        analysis,
        output_dir
        / "attention_by_token_position.png",
    )

    save_early_token_attention(
        analysis,
        output_dir
        / "early_token_attention.png",
    )

    save_layer_attention_heatmap(
        analysis,
        output_dir
        / "layer_attention_heatmap.png",
    )

    save_head_attention_heatmap(
        analysis,
        output_dir
        / "first_layer_head_attention.png",
    )

    print(
        "\nPlots written to:"
    )

    print(
        output_dir.resolve()
    )


if __name__ == "__main__":
    main()