import torch

from src.position_utils import (
    build_absolute_position_ids,
    build_cache_position,
    ensure_rope_cache_length,
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


class FakeRotaryEmbedding:
    def __init__(self):
        self.max_seq_len_cached = 8
        self.inv_freq = torch.ones(
            4,
            dtype=torch.float32,
        )
        self.calls = []

    def _set_cos_sin_cache(
        self,
        seq_len,
        device,
        dtype,
    ):
        self.calls.append(
            (
                seq_len,
                device,
                dtype,
            )
        )
        self.max_seq_len_cached = seq_len


class FakeAttention:
    def __init__(self):
        self.rotary_emb = FakeRotaryEmbedding()


class FakeLayer:
    def __init__(self):
        self.self_attn = FakeAttention()


class FakeBaseModel:
    def __init__(self):
        self.layers = [
            FakeLayer(),
        ]


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeBaseModel()
        self.weight = torch.nn.Parameter(
            torch.ones(1)
        )


def test_legacy_rope_cache_is_extended():
    model = FakeModel()

    ensure_rope_cache_length(
        model,
        required_length=32,
    )

    rotary = (
        model.model
        .layers[0]
        .self_attn
        .rotary_emb
    )

    assert rotary.max_seq_len_cached == 32
    assert len(rotary.calls) == 1
    assert rotary.calls[0][0] == 32


def test_rope_cache_is_not_rebuilt_when_large_enough():
    model = FakeModel()

    ensure_rope_cache_length(
        model,
        required_length=4,
    )

    rotary = (
        model.model
        .layers[0]
        .self_attn
        .rotary_emb
    )

    assert rotary.max_seq_len_cached == 8
    assert len(rotary.calls) == 0


if __name__ == "__main__":
    test_absolute_position_ids_continue_after_eviction()
    test_cache_positions_are_absolute()
    test_legacy_rope_cache_is_extended()
    test_rope_cache_is_not_rebuilt_when_large_enough()

    print("Stage 6 position-handling tests passed.")