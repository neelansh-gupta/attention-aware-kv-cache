"""Small real-model Sliding Window vs StreamingLLM experiment."""

from __future__ import annotations

import json
from pathlib import Path

from src.model_wrapper import ModelWrapper


def main() -> None:
    prompt = (
        "The system instruction is to answer concisely. "
        "A user discusses cache compression, attention, memory, latency, "
        "long documents, retrieval, and generation quality in detail. "
    )
    budget = 8
    max_new_tokens = 4
    wrapper = ModelWrapper(device="cpu")

    sliding = wrapper.sliding_window_generate(
        prompt,
        max_new_tokens=max_new_tokens,
        window_size=budget,
        return_diagnostics=True,
    )
    streaming = wrapper.streaming_llm_generate(
        prompt,
        max_new_tokens=max_new_tokens,
        cache_budget=budget,
        sink_tokens=2,
        return_diagnostics=True,
    )

    result = {
        "model": wrapper.model_name,
        "device": str(wrapper.device),
        "prompt_tokens": int(wrapper.tokenize(prompt)["input_ids"].shape[-1]),
        "cache_budget": budget,
        "max_new_tokens": max_new_tokens,
        "sliding": {
            "generation_succeeded": True,
            "cache_length": sliding.cache_length,
            "retained_positions": sliding.retained_positions,
            "generated_token_ids": sliding.generated_token_ids,
            "text": sliding.text,
        },
        "streaming": {
            "generation_succeeded": True,
            "sink_tokens": 2,
            "cache_length": streaming.cache_length,
            "retained_positions": streaming.retained_positions,
            "generated_token_ids": streaming.generated_token_ids,
            "text": streaming.text,
        },
    }
    assert sliding.cache_length <= budget
    assert streaming.cache_length <= budget
    assert len(set(streaming.retained_positions)) == streaming.cache_length

    output = Path("results/stage4_sliding_vs_streaming.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
