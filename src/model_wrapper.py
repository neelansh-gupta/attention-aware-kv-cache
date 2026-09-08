from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .cache_manager import SlidingWindowCacheManager


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

    Note:
        Correct RoPE/position handling after cache eviction is
        intentionally deferred to Stage 6.
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

        Important:
            Correct RoPE/position handling after eviction is
            intentionally deferred to Stage 6.
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