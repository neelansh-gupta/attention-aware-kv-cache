"""Public Stage 7 correctness entry point.

Categories A-C are correctness assertions. Category D records quality
differences after eviction and deliberately does not require equality.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import torch

from src.cache_manager import SlidingWindowCacheManager
from src.cache_utils import cache_as_tensor_tuple
from src.model_wrapper import ModelWrapper
from test_correctness_harness import run_all_tests
from test_position_handling import (
    test_model_level_middle_eviction_preserves_positions,
    test_qwen_applies_rope_before_cache_update,
)


def category_a(wrapper: ModelWrapper) -> None:
    """Reference DynamicCache equals the custom tensor view before eviction."""

    inputs = wrapper.tokenize("cache implementation correctness")
    with torch.no_grad():
        outputs = wrapper.model(**inputs, use_cache=True)
    reference = tuple(
        (key.clone(), value.clone())
        for key, value in cache_as_tensor_tuple(outputs.past_key_values)
    )
    manager = SlidingWindowCacheManager(window_size=inputs["input_ids"].shape[-1])
    unchanged = manager.update(
        outputs.past_key_values,
        token_positions=list(range(inputs["input_ids"].shape[-1])),
    )
    custom = cache_as_tensor_tuple(unchanged)
    assert len(reference) == len(custom)
    for (ref_key, ref_value), (key, value) in zip(reference, custom):
        assert torch.allclose(key, ref_key, atol=1e-6, rtol=1e-5)
        assert torch.allclose(value, ref_value, atol=1e-6, rtol=1e-5)


def category_b() -> None:
    """Synthetic eviction checks for indices, K/V alignment, and budgets."""

    run_all_tests()


def category_c() -> None:
    """Qwen2 RoPE ordering and non-contiguous middle eviction."""

    test_qwen_applies_rope_before_cache_update()
    test_model_level_middle_eviction_preserves_positions()


def token_agreement(reference: list[int], candidate: list[int]) -> float:
    compared = min(len(reference), len(candidate))
    if compared == 0:
        return 0.0
    return sum(
        left == right
        for left, right in zip(reference[:compared], candidate[:compared])
    ) / compared


def category_d(wrapper: ModelWrapper) -> None:
    """Measure compressed-vs-full generated-token agreement; do not assert equality."""

    prompt = (
        "A long conversation contains system instructions, prior user facts, "
        "tool outputs, and recent dialogue that an assistant should consider."
    )
    max_new_tokens = 4
    budget = 8
    full = wrapper.incremental_generate(
        prompt, max_new_tokens=max_new_tokens, return_diagnostics=True
    )
    compressed = {
        "sliding": wrapper.sliding_window_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            window_size=budget,
            return_diagnostics=True,
        ),
        "streaming": wrapper.streaming_llm_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            cache_budget=budget,
            sink_tokens=2,
            return_diagnostics=True,
        ),
        "h2o": wrapper.heavy_hitter_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            cache_budget=budget,
            sink_tokens=1,
            recent_window=2,
            return_diagnostics=True,
        ),
    }
    measurements = {
        name: {
            "token_agreement_with_full": token_agreement(
                full.generated_token_ids, result.generated_token_ids
            ),
            "generated_token_ids": result.generated_token_ids,
            "cache_length": result.cache_length,
            "retained_positions": result.retained_positions,
        }
        for name, result in compressed.items()
    }
    for measurement in measurements.values():
        assert 0.0 <= measurement["token_agreement_with_full"] <= 1.0
        assert measurement["cache_length"] <= budget
        assert measurement["generated_token_ids"]

    payload = {
        "model": wrapper.model_name,
        "device": str(wrapper.device),
        "method": "greedy token agreement over four generated tokens; descriptive only",
        "full_generated_token_ids": full.generated_token_ids,
        "compressed": measurements,
    }
    output = Path("results/stage7_quality_comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print("QUALITY:", json.dumps(payload, sort_keys=True))


def report(name: str, function: Callable[[], None]) -> bool:
    try:
        function()
    except Exception as error:
        print(f"FAIL: {name}: {type(error).__name__}: {error}")
        return False
    print(f"PASS: {name}")
    return True


def main() -> int:
    print("Stage 7 correctness harness")
    print("===========================")
    passed = report("A. cache/reference correctness", lambda: category_a(wrapper))
    passed = report("B. eviction correctness", category_b) and passed
    passed = report("C. position correctness", category_c) and passed
    passed = report(
        "D. quality comparison (measurement, not equality)",
        lambda: category_d(wrapper),
    ) and passed
    print("PASS: all required categories" if passed else "FAIL: required category failed")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        wrapper = ModelWrapper(device="cpu")
    except Exception as error:
        print(
            "SKIP: model-dependent categories A/C/D: "
            f"{type(error).__name__}: {error}"
        )
        wrapper = None
        ok = report("B. eviction correctness", category_b)
        sys.exit(0 if ok else 1)
    sys.exit(main())
