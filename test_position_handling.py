import inspect

import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

from src.cache_manager import AttentionSinkCacheManager
from src.cache_utils import cache_as_tensor_tuple
from src.model_wrapper import ModelWrapper
from src.position_utils import (
    build_absolute_position_ids,
    build_cache_position,
)


def test_absolute_position_ids_continue_after_eviction():
    device = torch.device("cpu")

    first_positions = build_absolute_position_ids(
        start_position=0,
        sequence_length=8,
        device=device,
    )

    assert first_positions.tolist() == [
        list(range(8))
    ]

    # Simulate cache eviction.
    # The physical cache may now contain only a few entries,
    # but the next token is still at its original position.
    next_positions = build_absolute_position_ids(
        start_position=8,
        sequence_length=1,
        device=device,
    )

    assert next_positions.tolist() == [[8]]

    later_positions = build_absolute_position_ids(
        start_position=20,
        sequence_length=1,
        device=device,
    )

    assert later_positions.tolist() == [[20]]


def test_cache_positions_are_absolute():
    device = torch.device("cpu")

    cache_position = build_cache_position(
        start_position=15,
        sequence_length=1,
        device=device,
    )

    assert cache_position.tolist() == [15]


def test_qwen_applies_rope_before_cache_update():
    source = inspect.getsource(Qwen2Attention.forward)
    assert source.index("apply_rotary_pos_emb") < source.index("past_key_values.update")


def test_model_level_middle_eviction_preserves_positions():
    wrapper = ModelWrapper(device="cpu")
    tokenized = wrapper.tokenize(
        "zero one two three four five six seven eight nine ten eleven"
    )
    input_ids = tokenized["input_ids"][:, :10]
    positions = torch.arange(10, device=wrapper.device).unsqueeze(0)
    with torch.no_grad():
        full = wrapper.model(
            input_ids=input_ids,
            position_ids=positions,
            use_cache=True,
        )
    full_layers = tuple(
        (key.clone(), value.clone())
        for key, value in cache_as_tensor_tuple(full.past_key_values)
    )

    manager = AttentionSinkCacheManager(cache_budget=6, sink_tokens=3)
    compressed = manager.update(
        full.past_key_values,
        token_positions=list(range(10)),
    )
    retained = torch.tensor([0, 1, 2, 7, 8, 9])
    assert manager.retained_positions.tolist() == retained.tolist()
    for (full_key, full_value), (key, value) in zip(
        full_layers, cache_as_tensor_tuple(compressed)
    ):
        assert torch.equal(key, full_key.index_select(-2, retained))
        assert torch.equal(value, full_value.index_select(-2, retained))

    captured_positions = []

    def capture_position_ids(module, args):
        captured_positions.append(args[1].detach().cpu().clone())

    hook = wrapper.model.model.rotary_emb.register_forward_pre_hook(
        capture_position_ids
    )
    next_token = tokenized["input_ids"][:, 10:11]
    with torch.no_grad():
        wrapper.model(
            input_ids=next_token,
            position_ids=torch.tensor([[10]], device=wrapper.device),
            past_key_values=compressed,
            use_cache=True,
        )
    hook.remove()
    assert captured_positions[-1].tolist() == [[10]]
    assert captured_positions[-1].tolist() != [[6]]


if __name__ == "__main__":
    test_absolute_position_ids_continue_after_eviction()
    test_cache_positions_are_absolute()
    test_qwen_applies_rope_before_cache_update()
    test_model_level_middle_eviction_preserves_positions()

    print("Stage 6 position-handling tests passed.")