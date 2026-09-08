from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B"


@dataclass
class CacheInfo:
    num_layers: int
    sequence_length: int
    key_shape: tuple
    value_shape: tuple


@dataclass
class AttentionAnalysis:
    """
    Container for attention analysis results.

    attention_by_layer:
        List of tensors, one per layer.
        Each tensor has shape:
            [num_heads, query_length, key_length]

    mean_attention_by_layer:
        List of tensors, one per layer.
        Each tensor has shape:
            [query_length, key_length]

    token_attention_received:
        Tensor containing attention received by each key/token position,
        averaged across layers, heads and query positions.
        Shape:
            [key_length]
    """

    attention_by_layer: list[torch.Tensor]
    mean_attention_by_layer: list[torch.Tensor]
    token_attention_received: torch.Tensor

    def early_token_attention(self, token_counts=(1, 2, 4, 8)) -> dict[int, float]:
        """
        Return the fraction of total received attention directed to
        the first N tokens.
        """
        results = {}

        total = self.token_attention_received.sum().item()

        if total <= 0:
            return {n: 0.0 for n in token_counts}

        for n in token_counts:
            n = min(n, self.token_attention_received.numel())

            mass = self.token_attention_received[:n].sum().item()

            results[n] = mass / total

        return results


class ModelWrapper:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
    ):
        self.model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        print(f"Loading tokenizer: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        print(f"Loading model on {self.device}")

        model_kwargs = {}

        # We explicitly use eager attention because we need access to
        # attention probability tensors.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="eager",
                **model_kwargs,
            )
        except TypeError:
            # Compatibility fallback for older transformers versions.
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs,
            )

        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> dict[str, torch.Tensor]:
        """
        Tokenize text and move tensors to the selected device.
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
    # Forward pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        text: str,
        output_attentions: bool = False,
        use_cache: bool = True,
    ):
        """
        Run a normal forward pass.

        output_attentions=True is required for attention analysis.
        """
        inputs = self.tokenize(text)

        outputs = self.model(
            **inputs,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        return outputs

    # ------------------------------------------------------------------
    # Cache inspection
    # ------------------------------------------------------------------

    def inspect_cache(
        self,
        past_key_values: Any,
    ) -> Optional[CacheInfo]:
        """
        Inspect both modern Hugging Face Cache objects and legacy
        tuple-style past_key_values.
        """

        if past_key_values is None:
            return None

        # --------------------------------------------------------------
        # Modern Hugging Face Cache API
        # --------------------------------------------------------------

        if hasattr(past_key_values, "key_cache"):
            key_cache = past_key_values.key_cache
            value_cache = past_key_values.value_cache

            if len(key_cache) == 0:
                return None

            key = key_cache[0]
            value = value_cache[0]

            sequence_length = key.shape[-2]

            return CacheInfo(
                num_layers=len(key_cache),
                sequence_length=sequence_length,
                key_shape=tuple(key.shape),
                value_shape=tuple(value.shape),
            )

        # --------------------------------------------------------------
        # Legacy tuple-style cache
        # --------------------------------------------------------------

        if isinstance(past_key_values, (tuple, list)):
            if len(past_key_values) == 0:
                return None

            first_layer = past_key_values[0]

            if isinstance(first_layer, (tuple, list)):
                key = first_layer[0]
                value = first_layer[1]

                sequence_length = key.shape[-2]

                return CacheInfo(
                    num_layers=len(past_key_values),
                    sequence_length=sequence_length,
                    key_shape=tuple(key.shape),
                    value_shape=tuple(value.shape),
                )

        return None

    # ------------------------------------------------------------------
    # Attention inspection
    # ------------------------------------------------------------------

    def inspect_attention(self, outputs) -> list[torch.Tensor]:
        """
        Extract attention tensors from a model output.

        Each tensor is expected to have shape:

            [batch, heads, query_length, key_length]

        The returned tensors have the batch dimension removed.
        """

        attentions = getattr(outputs, "attentions", None)

        if attentions is None:
            return []

        result = []

        for attention in attentions:
            if attention is None:
                continue

            # We analyze one example at a time.
            # Shape:
            # [batch, heads, query_length, key_length]
            if attention.dim() == 4:
                attention = attention[0]

            result.append(
                attention.detach().float().cpu()
            )

        return result

    # ------------------------------------------------------------------
    # Attention analysis
    # ------------------------------------------------------------------

    def analyze_attention(
        self,
        text: str,
    ) -> AttentionAnalysis:
        """
        Run the model with attention outputs enabled and calculate:

        1. Attention tensors for every layer.
        2. Mean attention over heads for every layer.
        3. Attention received by every token position.

        The analysis is performed on the supplied text.

        No attention-sink conclusion is hard-coded here.
        """

        outputs = self.forward(
            text,
            output_attentions=True,
            use_cache=False,
        )

        attentions = self.inspect_attention(outputs)

        if not attentions:
            raise RuntimeError(
                "No attention tensors were returned. "
                "Make sure eager attention is enabled and "
                "output_attentions=True is supported."
            )

        mean_attention_by_layer = []

        # --------------------------------------------------------------
        # Average attention across heads.
        #
        # Original:
        # [heads, query_length, key_length]
        #
        # Result:
        # [query_length, key_length]
        # --------------------------------------------------------------

        for layer_attention in attentions:
            mean_attention = layer_attention.mean(dim=0)

            mean_attention_by_layer.append(
                mean_attention
            )

        # --------------------------------------------------------------
        # Calculate attention RECEIVED by each token.
        #
        # For each layer:
        #
        #   average over heads
        #   average over query positions
        #
        # Result:
        #   [key_length]
        # --------------------------------------------------------------

        received_attention_per_layer = []

        for layer_attention in attentions:
            mean_over_heads = layer_attention.mean(dim=0)

            received = mean_over_heads.mean(dim=0)

            received_attention_per_layer.append(
                received
            )

        token_attention_received = torch.stack(
            received_attention_per_layer,
            dim=0,
        ).mean(dim=0)

        return AttentionAnalysis(
            attention_by_layer=attentions,
            mean_attention_by_layer=mean_attention_by_layer,
            token_attention_received=token_attention_received,
        )

    # ------------------------------------------------------------------
    # Early-token sink analysis
    # ------------------------------------------------------------------

    def analyze_attention_sinks(
        self,
        text: str,
        token_counts=(1, 2, 4, 8),
    ) -> dict[str, Any]:
        """
        Calculate statistics describing how much attention is received
        by early tokens.

        This function intentionally reports measurements rather than
        declaring that attention sinks exist.
        """

        analysis = self.analyze_attention(text)

        early_mass = analysis.early_token_attention(
            token_counts=token_counts
        )

        received = analysis.token_attention_received

        if received.numel() == 0:
            raise RuntimeError(
                "Attention analysis produced no token positions."
            )

        # Most-attended token position.
        top_position = int(
            torch.argmax(received).item()
        )

        # Normalize to a probability distribution.
        normalized_received = (
            received / received.sum()
        )

        return {
            "sequence_length": int(received.numel()),
            "early_token_attention": early_mass,
            "most_attended_token_position": top_position,
            "token_attention_received": normalized_received,
            "analysis": analysis,
        }

    # ------------------------------------------------------------------
    # Incremental generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def incremental_generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
    ) -> str:
        """
        Generate tokens one at a time while explicitly carrying the
        past_key_values cache.
        """

        inputs = self.tokenize(prompt)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

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
                [generated_ids, next_token],
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

            # Stop at EOS.
            if self.tokenizer.eos_token_id is not None:
                if torch.all(
                    next_token
                    == self.tokenizer.eos_token_id
                ):
                    break

        return self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

    # ------------------------------------------------------------------
    # Model summary
    # ------------------------------------------------------------------

    def print_model_summary(self):
        config = self.model.config

        print("\nModel Summary")
        print("-------------")
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")

        for attribute in [
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "hidden_size",
            "max_position_embeddings",
        ]:
            value = getattr(config, attribute, "N/A")
            print(f"{attribute}: {value}")

    # ------------------------------------------------------------------
    # Stage 1 smoke test
    # ------------------------------------------------------------------

    def run_smoke_test(self):
        """
        Basic Stage 1 compatibility test.
        """

        prompt = "The capital of France is"

        outputs = self.forward(
            prompt,
            output_attentions=True,
            use_cache=True,
        )

        print("Forward pass successful.")

        cache_info = self.inspect_cache(
            outputs.past_key_values
        )

        if cache_info is not None:
            print("\nKV Cache")
            print("--------")
            print(f"Layers: {cache_info.num_layers}")
            print(
                f"Sequence length: "
                f"{cache_info.sequence_length}"
            )
            print(
                f"Key shape: "
                f"{cache_info.key_shape}"
            )
            print(
                f"Value shape: "
                f"{cache_info.value_shape}"
            )

        attentions = self.inspect_attention(outputs)

        print("\nAttention")
        print("---------")
        print(f"Number of layers: {len(attentions)}")

        if attentions:
            print(
                f"First layer attention shape: "
                f"{tuple(attentions[0].shape)}"
            )

        generated = self.incremental_generate(
            prompt,
            max_new_tokens=16,
        )

        print("\nGenerated text:")
        print(generated)


def main():
    wrapper = ModelWrapper()

    wrapper.print_model_summary()

    wrapper.run_smoke_test()


if __name__ == "__main__":
    main()