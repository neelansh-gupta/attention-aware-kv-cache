from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .cache_manager import (
    AttentionSinkCacheManager,
    HeavyHitterCacheManager,
    SlidingWindowCacheManager,
)

from .position_utils import (
    build_absolute_position_ids,
    build_cache_position,
    ensure_rope_cache_length,
)


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B"


@dataclass
class CacheInfo:
    """
    Basic information about the model KV cache.
    """

    num_layers: int
    sequence_length: int
    key_shape: tuple
    value_shape: tuple


@dataclass
class AttentionAnalysis:
    """
    Container for attention-analysis results.
    """

    attention_received: torch.Tensor
    early_token_attention: torch.Tensor
    most_attended_token: int


class ModelWrapper:
    """
    Wrapper around a Hugging Face causal language model.

    Stages implemented:
        Stage 1:
            - Model loading
            - Tokenization
            - Incremental generation
            - KV-cache inspection

        Stage 2:
            - Attention extraction
            - Attention aggregation
            - Attention-sink analysis

        Stage 3:
            - Sliding-window KV-cache eviction

        Stage 4:
            - StreamingLLM-style attention-sink-aware KV-cache eviction

        Stage 5:
            - H2O-style accumulated-attention heavy-hitter KV-cache eviction

        Stage 6:
            - Absolute position tracking after KV-cache eviction
            - Correct position_ids for continued generation
            - RoPE cache extension for legacy implementations
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
    ):
        self.model_name = model_name

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
        )

        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def tokenize(
        self,
        text: str,
    ) -> dict[str, torch.Tensor]:
        """
        Tokenize input text and move tensors to the model device.
        """

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        )

        return {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

    # ------------------------------------------------------------------
    # Stage 1: Incremental generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def incremental_generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
    ) -> str:
        """
        Generate tokens incrementally using the model's KV cache.
        """

        inputs = self.tokenize(prompt)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        generated_ids = input_ids.clone()

        past_key_values = None

        for _ in range(max_new_tokens):

            if past_key_values is None:
                current_input_ids = input_ids
            else:
                current_input_ids = generated_ids[:, -1:]

            position_ids = (
                attention_mask.cumsum(dim=-1) - 1
            )

            if past_key_values is not None:
                position_ids = position_ids[:, -1:]

            kwargs = {
                "input_ids": current_input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
            }

            try:
                outputs = self.model(
                    **kwargs,
                    cache_position=torch.arange(
                        attention_mask.shape[-1]
                        - current_input_ids.shape[-1],
                        attention_mask.shape[-1],
                        device=self.device,
                    ),
                )
            except (TypeError, ValueError):
                outputs = self.model(**kwargs)

            next_token = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            generated_ids = torch.cat(
                [
                    generated_ids,
                    next_token,
                ],
                dim=-1,
            )

            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=self.device,
                    ),
                ],
                dim=-1,
            )

            past_key_values = outputs.past_key_values

            if (
                self.tokenizer.eos_token_id is not None
                and torch.all(
                    next_token
                    == self.tokenizer.eos_token_id
                )
            ):
                break

        return self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

    # ------------------------------------------------------------------
    # Cache inspection
    # ------------------------------------------------------------------

    def inspect_cache(
        self,
        past_key_values: Any,
    ) -> Optional[CacheInfo]:
        """
        Inspect the structure and dimensions of a KV cache.
        """

        if past_key_values is None:
            return None

        legacy_cache = self._cache_to_legacy(
            past_key_values
        )

        if legacy_cache is None or len(legacy_cache) == 0:
            return CacheInfo(
                num_layers=0,
                sequence_length=0,
                key_shape=(),
                value_shape=(),
            )

        first_key, first_value = legacy_cache[0]

        return CacheInfo(
            num_layers=len(legacy_cache),
            sequence_length=first_key.shape[-2],
            key_shape=tuple(first_key.shape),
            value_shape=tuple(first_value.shape),
        )

    # ------------------------------------------------------------------
    # Stage 2: Attention extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inspect_attention(
        self,
        prompt: str,
    ) -> tuple[
        torch.Tensor,
        list[tuple[int, int]],
    ]:
        """
        Run the model with attentions enabled.

        Returns:
            attentions:
                Tuple-like structure containing attention tensors
                for every layer.

            token_ids:
                Token IDs corresponding to the input sequence.
        """

        inputs = self.tokenize(prompt)

        outputs = self.model(
            **inputs,
            output_attentions=True,
            use_cache=False,
        )

        attentions = outputs.attentions

        token_ids = inputs["input_ids"][0].tolist()

        return (
            attentions,
            [
                (token_id, index)
                for index, token_id in enumerate(token_ids)
            ],
        )

    def analyze_attention(
        self,
        attentions,
    ) -> AttentionAnalysis:
        """
        Aggregate attention across layers and heads.

        For each layer:
            [batch, heads, query, key]

        First average over heads, then average over layers.

        The resulting matrix has shape:
            [sequence_length, sequence_length]
        """

        if attentions is None or len(attentions) == 0:
            raise ValueError(
                "No attention tensors were provided."
            )

        layer_attention = []

        for attention in attentions:

            if attention.dim() != 4:
                raise ValueError(
                    "Expected attention tensor with shape "
                    "[batch, heads, query, key]."
                )

            averaged_heads = attention.mean(
                dim=1
            )

            layer_attention.append(
                averaged_heads
            )

        attention_matrix = torch.stack(
            layer_attention,
            dim=0,
        ).mean(dim=0)

        attention_matrix = attention_matrix[0]

        attention_received = attention_matrix.sum(
            dim=0
        )

        early_count = min(
            5,
            attention_matrix.shape[-1],
        )

        early_token_attention = attention_matrix[
            :,
            :early_count,
        ].sum(
            dim=0
        )

        most_attended_token = int(
            torch.argmax(
                attention_received
            ).item()
        )

        return AttentionAnalysis(
            attention_received=attention_received,
            early_token_attention=early_token_attention,
            most_attended_token=most_attended_token,
        )

    def analyze_attention_sinks(
        self,
        attentions,
        early_token_count: int = 5,
    ) -> dict[str, Any]:
        """
        Analyze how much attention is received by early tokens.

        Returns:
            A dictionary containing:
                - attention_received
                - early_token_attention
                - early_token_fraction
                - most_attended_token
        """

        analysis = self.analyze_attention(
            attentions
        )

        sequence_length = (
            analysis.attention_received.shape[-1]
        )

        early_token_count = min(
            early_token_count,
            sequence_length,
        )

        early_attention = (
            analysis.attention_received[
                :early_token_count
            ]
        )

        total_attention = (
            analysis.attention_received.sum()
        )

        if total_attention.item() == 0:
            early_fraction = torch.tensor(
                0.0,
                device=total_attention.device,
            )
        else:
            early_fraction = (
                early_attention.sum()
                / total_attention
            )

        return {
            "attention_received": (
                analysis.attention_received
            ),
            "early_token_attention": (
                early_attention
            ),
            "early_token_fraction": (
                early_fraction
            ),
            "most_attended_token": (
                analysis.most_attended_token
            ),
        }

    # ------------------------------------------------------------------
    # Stage 3: Sliding-window generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sliding_window_generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        window_size: int = 128,
    ) -> str:
        """
        Generate tokens while enforcing a fixed-size
        sliding-window KV-cache budget.

        Stage 3:
            - Uses Hugging Face's native DynamicCache.
            - Keeps only the most recent `window_size` entries.
            - Does not convert the cache to the legacy tuple format.

        Stage 6:
            - Tracks absolute token positions independently
              from the physical cache length.
            - Continues using the original sequence positions
              after cache eviction.
        """

        if window_size <= 0:
            raise ValueError(
                "window_size must be positive."
            )

        inputs = self.tokenize(prompt)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        manager = SlidingWindowCacheManager(
            window_size=window_size
        )

        past_key_values = None
        generated_ids = input_ids.clone()

        # Absolute position in the original sequence.
        #
        # This must not shrink when the physical KV cache
        # is cropped.
        next_position = 0

        for _ in range(max_new_tokens):

            if past_key_values is None:
                current_input_ids = input_ids
            else:
                current_input_ids = generated_ids[:, -1:]

            if past_key_values is None:

                position_ids = build_absolute_position_ids(
                    start_position=0,
                    sequence_length=current_input_ids.shape[-1],
                    device=self.device,
                )

                cache_position = build_cache_position(
                    start_position=0,
                    sequence_length=current_input_ids.shape[-1],
                    device=self.device,
                )

            else:

                position_ids = build_absolute_position_ids(
                    start_position=next_position,
                    sequence_length=current_input_ids.shape[-1],
                    device=self.device,
                )

                cache_position = build_cache_position(
                    start_position=next_position,
                    sequence_length=current_input_ids.shape[-1],
                    device=self.device,
                )

            ensure_rope_cache_length(
                self.model,
                int(position_ids.max().item()) + 1,
            )

            kwargs = {
                "input_ids": current_input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
            }

            try:
                outputs = self.model(
                    **kwargs,
                    cache_position=cache_position,
                )
            except (TypeError, ValueError):
                outputs = self.model(**kwargs)

            next_token = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            generated_ids = torch.cat(
                [
                    generated_ids,
                    next_token,
                ],
                dim=-1,
            )

            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=self.device,
                    ),
                ],
                dim=-1,
            )

            # The next generated token belongs to the next
            # absolute position in the original sequence.
            next_position += current_input_ids.shape[-1]

            # Keep the native Hugging Face DynamicCache.
            past_key_values = outputs.past_key_values

            # Enforce the sliding-window budget using
            # DynamicCache.crop().
            past_key_values = manager.update(
                past_key_values
            )

            # The mask must correspond to the retained cache
            # plus the current/new token.
            cache_length = manager.sequence_length(
                past_key_values
            )

            expected_mask_length = cache_length + 1

            if attention_mask.shape[-1] > expected_mask_length:
                attention_mask = attention_mask[
                    :,
                    -expected_mask_length:,
                ]

            if (
                self.tokenizer.eos_token_id is not None
                and torch.all(
                    next_token
                    == self.tokenizer.eos_token_id
                )
            ):
                break

        return self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

    # ------------------------------------------------------------------
    # Stage 4: StreamingLLM / attention-sink-aware generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def streaming_llm_generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        cache_budget: int = 128,
        sink_tokens: int = 4,
    ) -> str:
        """
        Generate tokens using a StreamingLLM-style KV-cache policy.

        The cache retains:
            - the first `sink_tokens` tokens
            - the most recent tokens filling the remaining budget

        Stage 6:
            Absolute position IDs continue increasing after
            cache eviction instead of being recomputed from
            the shortened attention mask.
        """

        if cache_budget <= 0:
            raise ValueError(
                "cache_budget must be positive."
            )

        if sink_tokens < 0:
            raise ValueError(
                "sink_tokens must be non-negative."
            )

        if sink_tokens >= cache_budget:
            raise ValueError(
                "sink_tokens must be smaller than cache_budget."
            )

        inputs = self.tokenize(prompt)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        manager = AttentionSinkCacheManager(
            cache_budget=cache_budget,
            sink_tokens=sink_tokens,
        )

        past_key_values = None
        generated_ids = input_ids.clone()

        # Absolute position in the original sequence.
        next_position = 0

        for _ in range(max_new_tokens):

            if past_key_values is None:
                current_input_ids = input_ids
            else:
                current_input_ids = generated_ids[:, -1:]

            if past_key_values is None:

                position_ids = build_absolute_position_ids(
                    start_position=0,
                    sequence_length=current_input_ids.shape[-1],
                    device=self.device,
                )

                cache_position = build_cache_position(
                    start_position=0,
                    sequence_length=current_input_ids.shape[-1],
                    device=self.device,
                )

            else:

                position_ids = build_absolute_position_ids(
                    start_position=next_position,
                    sequence_length=current_input_ids.shape[-1],
                    device=self.device,
                )

                cache_position = build_cache_position(
                    start_position=next_position,
                    sequence_length=current_input_ids.shape[-1],
                    device=self.device,
                )

            ensure_rope_cache_length(
                self.model,
                int(position_ids.max().item()) + 1,
            )

            kwargs = {
                "input_ids": current_input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
            }

            try:
                outputs = self.model(
                    **kwargs,
                    cache_position=cache_position,
                )
            except (TypeError, ValueError):
                outputs = self.model(**kwargs)

            next_token = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            generated_ids = torch.cat(
                [
                    generated_ids,
                    next_token,
                ],
                dim=-1,
            )

            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=self.device,
                    ),
                ],
                dim=-1,
            )

            # Continue absolute positions independently
            # from the physical cache length.
            next_position += current_input_ids.shape[-1]

            past_key_values = outputs.past_key_values

            past_key_values = manager.update(
                past_key_values
            )

            cache_length = manager.sequence_length(
                past_key_values
            )

            expected_mask_length = cache_length + 1

            if attention_mask.shape[-1] > expected_mask_length:
                attention_mask = attention_mask[
                    :,
                    -expected_mask_length:,
                ]

            if (
                self.tokenizer.eos_token_id is not None
                and torch.all(
                    next_token
                    == self.tokenizer.eos_token_id
                )
            ):
                break

        return self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

    # ------------------------------------------------------------------
    # Stage 5: H2O / heavy-hitter generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def heavy_hitter_generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        cache_budget: int = 128,
    ) -> str:
        """
        Generate tokens using an H2O-style heavy-hitter KV-cache policy.

        At every decoding step, attention received by cached tokens
        is accumulated. When the cache exceeds the configured budget,
        the tokens with the highest accumulated attention scores are
        retained.

        Stage 6:
            Absolute position IDs are preserved across
            non-contiguous heavy-hitter eviction.
        """

        if cache_budget <= 0:
            raise ValueError(
                "cache_budget must be positive."
            )

        inputs = self.tokenize(prompt)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        manager = HeavyHitterCacheManager(
            cache_budget=cache_budget
        )

        past_key_values = None
        generated_ids = input_ids.clone()

        accumulated_scores = None

        # Absolute position in the original sequence.
        #
        # Heavy-hitter eviction may remove arbitrary tokens,
        # so this value must remain independent of the
        # physical cache length.
        next_position = 0

        for _ in range(max_new_tokens):

            if past_key_values is None:
                current_input_ids = input_ids
            else:
                current_input_ids = (
                    generated_ids[:, -1:]
                )

            if past_key_values is None:

                position_ids = build_absolute_position_ids(
                    start_position=0,
                    sequence_length=current_input_ids.shape[-1],
                    device=self.device,
                )

                cache_position = build_cache_position(
                    start_position=0,
                    sequence_length=current_input_ids.shape[-1],
                    device=self.device,
                )

            else:

                position_ids = build_absolute_position_ids(
                    start_position=next_position,
                    sequence_length=current_input_ids.shape[-1],
                    device=self.device,
                )

                cache_position = build_cache_position(
                    start_position=next_position,
                    sequence_length=current_input_ids.shape[-1],
                    device=self.device,
                )

            ensure_rope_cache_length(
                self.model,
                int(position_ids.max().item()) + 1,
            )

            kwargs = {
                "input_ids": current_input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
                "output_attentions": True,
            }

            try:
                outputs = self.model(
                    **kwargs,
                    cache_position=cache_position,
                )
            except (TypeError, ValueError):
                outputs = self.model(**kwargs)

            next_token = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            generated_ids = torch.cat(
                [
                    generated_ids,
                    next_token,
                ],
                dim=-1,
            )

            # ----------------------------------------------------------
            # Accumulate attention received by each cached token.
            # ----------------------------------------------------------

            if outputs.attentions is None:
                raise RuntimeError(
                    "Stage 5 requires attention tensors, but "
                    "the model did not return them."
                )

            current_attention = []

            for layer_attention in outputs.attentions:

                if layer_attention.dim() != 4:
                    raise ValueError(
                        "Expected attention tensor with shape "
                        "[batch, heads, query, key]."
                    )

                # We only need the attention generated by the
                # newest query token.
                newest_attention = (
                    layer_attention[:, :, -1, :]
                )

                # Average across attention heads.
                newest_attention = (
                    newest_attention.mean(dim=1)
                )

                # Batch size is one for this generation path.
                newest_attention = (
                    newest_attention[0]
                )

                current_attention.append(
                    newest_attention
                )

            # Average attention across layers.
            current_attention = torch.stack(
                current_attention,
                dim=0,
            ).mean(dim=0)

            # ----------------------------------------------------------
            # The attention vector includes the newly generated token.
            # ----------------------------------------------------------

            if accumulated_scores is None:

                accumulated_scores = (
                    current_attention.clone()
                )

            else:

                previous_length = (
                    accumulated_scores.shape[0]
                )

                if current_attention.shape[0] == (
                    previous_length + 1
                ):

                    # Existing cached tokens receive their
                    # newly accumulated attention.
                    accumulated_scores = (
                        accumulated_scores
                        + current_attention[
                            :previous_length
                        ]
                    )

                    # The newly generated token starts with
                    # the attention it received at this step.
                    accumulated_scores = torch.cat(
                        [
                            accumulated_scores,
                            current_attention[
                                previous_length:
                            ],
                        ],
                        dim=0,
                    )

                elif current_attention.shape[0] == previous_length:

                    # Some Hugging Face cache implementations may
                    # expose attention only over the existing cache.
                    accumulated_scores = (
                        accumulated_scores
                        + current_attention
                    )

                else:
                    raise ValueError(
                        "Attention sequence length does not "
                        "match the tracked KV-cache length."
                    )

            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=self.device,
                    ),
                ],
                dim=-1,
            )

            # Continue absolute positions independently
            # from the shortened physical cache.
            next_position += current_input_ids.shape[-1]

            past_key_values = outputs.past_key_values

            # ----------------------------------------------------------
            # Enforce the heavy-hitter cache budget.
            # ----------------------------------------------------------

            past_key_values, accumulated_scores = (
                manager.update(
                    past_key_values,
                    accumulated_scores,
                )
            )

            cache_length = manager.sequence_length(
                past_key_values
            )

            expected_mask_length = (
                cache_length + 1
            )

            if (
                attention_mask.shape[-1]
                > expected_mask_length
            ):
                attention_mask = attention_mask[
                    :,
                    -expected_mask_length:,
                ]

            if (
                self.tokenizer.eos_token_id is not None
                and torch.all(
                    next_token
                    == self.tokenizer.eos_token_id
                )
            ):
                break

        return self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

def main():
    """
    Basic Stage 1/Stage 2 model-wrapper smoke test.
    """

    wrapper = ModelWrapper()

    prompt = (
        "The history of artificial intelligence is"
    )

    print("Model:", wrapper.model_name)
    print("Device:", wrapper.device)

    print("\nIncremental generation:")

    result = wrapper.incremental_generate(
        prompt,
        max_new_tokens=16,
    )

    print(result)

    print("\nAttention analysis:")

    attentions, token_info = (
        wrapper.inspect_attention(
            prompt
        )
    )

    analysis = wrapper.analyze_attention(
        attentions
    )

    print(
        "Most attended token index:",
        analysis.most_attended_token,
    )

    print(
        "Attention received:",
        analysis.attention_received,
    )

    print("\nAttention sink analysis:")

    sink_analysis = (
        wrapper.analyze_attention_sinks(
            attentions
        )
    )

    print(
        "Early-token attention fraction:",
        sink_analysis[
            "early_token_fraction"
        ].item(),
    )


if __name__ == "__main__":
    main()