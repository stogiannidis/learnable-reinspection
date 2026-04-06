"""InternVL3 with Re-Inspection Module.

Wraps `InternVLForConditionalGeneration`, inserting task-conditioned
re-inspection tokens (R) into the prompt right before generation, or before the
first supervised answer token during training.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import AutoProcessor, InternVLForConditionalGeneration

from reinspection_vlm.config import ReInspectionConfig
from reinspection_vlm.outputs import ReInspectionOutput
from reinspection_vlm.reinspection_module import ReInspectionModule


def _chunked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Compute cross-entropy in chunks, upcasting each chunk to FP32.

    Avoids materialising the full (B*T, V) tensor in FP32 at once, which
    can easily exceed GPU memory for large-vocab models like InternVL3 (152K).
    """
    flat_logits = logits.view(-1, logits.size(-1))
    flat_labels = labels.view(-1)

    valid = flat_labels != ignore_index
    n_valid = valid.sum()
    if n_valid == 0:
        return flat_logits.sum() * 0.0

    total_loss = torch.zeros((), device=logits.device, dtype=torch.float32)
    n = flat_logits.size(0)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_labels = flat_labels[start:end]
        mask = chunk_labels != ignore_index
        if not mask.any():
            continue
        chunk_loss = torch.nn.functional.cross_entropy(
            flat_logits[start:end].float(),
            chunk_labels,
            ignore_index=ignore_index,
            reduction="sum",
        )
        total_loss = total_loss + chunk_loss
    return total_loss / n_valid.float()


class InternVL3WithReInspection(nn.Module):
    """InternVL3-8B + Re-Inspection Module."""

    def __init__(
        self,
        config: ReInspectionConfig,
        base_model: InternVLForConditionalGeneration,
        processor=None,
    ):
        super().__init__()
        self.config = config
        self.base_model = base_model
        self.reinspection = ReInspectionModule(config)
        self._image_token_id = base_model.config.image_token_id
        self._pad_token_id = getattr(base_model.config.text_config, "pad_token_id", 0) or 0

        self._last_attn_task = None
        self._last_attn_vis = None
        self._last_generation_prompt_lengths = None
        self._assistant_generation_suffix = self._resolve_generation_suffix(processor)

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    @property
    def dtype(self):
        return next(self.base_model.parameters()).dtype

    def _resolve_generation_suffix(self, processor) -> Optional[torch.LongTensor]:
        tokenizer = getattr(processor, "tokenizer", None) if processor is not None else None
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            return None

        messages = [{"role": "user", "content": "ping"}]
        try:
            with_prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            without_prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        except Exception:
            return None

        if isinstance(with_prompt, dict):
            with_prompt = with_prompt["input_ids"]
        if isinstance(without_prompt, dict):
            without_prompt = without_prompt["input_ids"]
        if with_prompt and isinstance(with_prompt[0], list):
            with_prompt = with_prompt[0]
        if without_prompt and isinstance(without_prompt[0], list):
            without_prompt = without_prompt[0]

        suffix = with_prompt[len(without_prompt) :]
        if not suffix:
            return None
        return torch.tensor(suffix, dtype=torch.long)

    def _sequence_lengths(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.LongTensor:
        if attention_mask is not None:
            return attention_mask.long().sum(dim=-1)
        return torch.full(
            (input_ids.shape[0],),
            input_ids.shape[1],
            dtype=torch.long,
            device=input_ids.device,
        )

    def _find_insert_positions(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> torch.LongTensor:
        """Find where re-inspection tokens should be inserted."""
        lengths = self._sequence_lengths(input_ids, attention_mask)
        positions = lengths.clone()

        if labels is not None:
            for b in range(input_ids.shape[0]):
                supervised = (labels[b] != self.config.answer_ignore_index).nonzero(as_tuple=False).squeeze(-1)
                if supervised.numel() > 0:
                    positions[b] = supervised[0]
            return positions

        if self._assistant_generation_suffix is None:
            return positions

        suffix = self._assistant_generation_suffix.to(input_ids.device)
        suffix_len = suffix.shape[0]
        for b in range(input_ids.shape[0]):
            seq_len = lengths[b].item()
            if seq_len >= suffix_len:
                tail = input_ids[b, seq_len - suffix_len : seq_len]
                if torch.equal(tail, suffix):
                    positions[b] = seq_len - suffix_len

        return positions

    def _extract_vision_and_text(
        self,
        inputs_embeds: torch.FloatTensor,
        input_ids: torch.LongTensor,
        insert_positions: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.BoolTensor, torch.BoolTensor]:
        """Extract padded vision/text tensors plus their validity masks."""
        B, L, D = inputs_embeds.shape

        vision_mask = input_ids == self._image_token_id
        if attention_mask is not None:
            vision_mask = vision_mask & attention_mask.bool()

        text_mask = ~vision_mask
        if attention_mask is not None:
            text_mask = text_mask & attention_mask.bool()

        V_list = []
        T_list = []
        v_counts = []
        t_counts = []
        for b in range(B):
            insert_pos = insert_positions[b].item()
            vis_positions = vision_mask[b].nonzero(as_tuple=False).squeeze(-1)

            text_mask_b = text_mask[b].clone()
            text_mask_b[insert_pos:] = False
            text_positions = text_mask_b.nonzero(as_tuple=False).squeeze(-1)

            v_counts.append(int(vis_positions.numel()))
            t_counts.append(int(text_positions.numel()))
            V_list.append(inputs_embeds[b, vis_positions] if vis_positions.numel() else inputs_embeds.new_zeros(1, D))
            T_list.append(inputs_embeds[b, text_positions] if text_positions.numel() else inputs_embeds.new_zeros(1, D))

        max_v = max(v.shape[0] for v in V_list) if V_list else 1
        max_t = max(t.shape[0] for t in T_list) if T_list else 1

        V = torch.zeros(B, max_v, D, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        T = torch.zeros(B, max_t, D, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        V_mask = torch.zeros(B, max_v, dtype=torch.bool, device=inputs_embeds.device)
        T_mask = torch.zeros(B, max_t, dtype=torch.bool, device=inputs_embeds.device)

        for b in range(B):
            nv = V_list[b].shape[0]
            nt = T_list[b].shape[0]
            V[b, :nv] = V_list[b]
            T[b, :nt] = T_list[b]
            if v_counts[b] > 0:
                V_mask[b, :v_counts[b]] = True
            else:
                V_mask[b, 0] = True
            if t_counts[b] > 0:
                T_mask[b, :t_counts[b]] = True
            else:
                T_mask[b, 0] = True

        return V, T, V_mask, T_mask

    def _insert_tokens(
        self,
        inputs_embeds: torch.FloatTensor,
        R: torch.FloatTensor,
        insert_positions: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        labels: Optional[torch.LongTensor],
        input_ids: Optional[torch.LongTensor],
    ) -> dict:
        """Insert re-inspection tokens and extend sequence-aligned tensors.

        Uses in-place assignment into a fresh zeros buffer (requires_grad=False),
        which detaches R from the frozen-LLM backward path. This prevents CE
        gradients from amplifying back through frozen bf16 layers and corrupting
        the trainable parameters. The reinspection module is trained in stage 1
        with attn supervision; in stage 2 only LoRA receives CE gradients.
        """
        B, L, D = inputs_embeds.shape
        N_q = R.shape[1]

        # --- embeddings: in-place into new buffer (detaches autograd through R) ---
        new_embeds = torch.zeros(
            B, L + N_q, D, device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        for b in range(B):
            pos = insert_positions[b].item()
            new_embeds[b, :pos] = inputs_embeds[b, :pos]
            new_embeds[b, pos:pos + N_q] = R[b]
            new_embeds[b, pos + N_q:] = inputs_embeds[b, pos:]

        # --- helper for non-differentiable integer/bool tensors ---
        def _insert_1d(src, fill_val, positions):
            parts = []
            for b in range(B):
                pos = positions[b].item()
                filler = src.new_full((N_q,), fill_val)
                parts.append(torch.cat([src[b, :pos], filler, src[b, pos:]], dim=0))
            return torch.stack(parts, dim=0)

        new_attention_mask = None
        if attention_mask is not None:
            new_attention_mask = _insert_1d(attention_mask, 1, insert_positions)

        new_labels = None
        if labels is not None:
            new_labels = _insert_1d(labels, self.config.answer_ignore_index, insert_positions)

        new_input_ids = None
        if input_ids is not None:
            new_input_ids = _insert_1d(input_ids, self._pad_token_id, insert_positions)

        new_position_ids = None
        if position_ids is not None:
            if position_ids.ndim == 2:
                pos_parts = []
                for b in range(B):
                    pos = insert_positions[b].item()
                    last_pos = position_ids[b, pos - 1].item() if pos > 0 else -1
                    r_pos = torch.arange(N_q, device=position_ids.device, dtype=position_ids.dtype) + last_pos + 1
                    pos_parts.append(
                        torch.cat([position_ids[b, :pos], r_pos, position_ids[b, pos:] + N_q], dim=0)
                    )
                new_position_ids = torch.stack(pos_parts, dim=0)
            else:
                raise ValueError(f"Unsupported position_ids rank: {position_ids.ndim}")

        return {
            "inputs_embeds": new_embeds,
            "attention_mask": new_attention_mask,
            "position_ids": new_position_ids,
            "labels": new_labels,
            "input_ids": new_input_ids,
        }

    def _nan_check(self, tensor: torch.Tensor, name: str, step: int) -> None:
        if tensor is not None and torch.isnan(tensor).any():
            raise RuntimeError(f"NaN detected in {name} at step {step}")

    def _encode_vision_and_scatter(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        pixel_values: Optional[torch.Tensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
    ) -> Tuple[torch.FloatTensor, Optional[torch.Tensor]]:
        if pixel_values is not None:
            image_outputs = self.base_model.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                return_dict=True,
            )
            image_features = image_outputs.pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = self.base_model.model.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
            return inputs_embeds, image_features
        return inputs_embeds, None

    def _prepare_reinspection_inputs(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        labels: Optional[torch.LongTensor],
        pixel_values: Optional[torch.Tensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
    ) -> dict:
        step = getattr(self, "_fwd_step", 0)

        if inputs_embeds is None:
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        self._nan_check(inputs_embeds, "token_embeds", step)

        inputs_embeds, image_features = self._encode_vision_and_scatter(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
        )
        self._nan_check(inputs_embeds, "embeds_after_vision_scatter", step)
        if image_features is not None:
            self._nan_check(image_features, "image_features", step)

        insert_positions = self._find_insert_positions(input_ids, attention_mask=attention_mask, labels=labels)
        V, T, V_mask, T_mask = self._extract_vision_and_text(
            inputs_embeds,
            input_ids,
            insert_positions,
            attention_mask=attention_mask,
        )
        self._nan_check(V, "V_extracted", step)
        self._nan_check(T, "T_extracted", step)

        R, A_task, A_vis, R_r = self.reinspection(
            V, T, V_mask=V_mask, T_mask=T_mask, need_weights=True,
        )
        self._nan_check(R, "R_tokens", step)

        self._last_attn_task = A_task.detach()
        self._last_attn_vis = A_vis.detach()

        inserted = self._insert_tokens(
            inputs_embeds, R, insert_positions,
            attention_mask, position_ids, labels, input_ids,
        )

        return {
            **inserted,
            "image_features": image_features,
            "A_task": A_task,
            "A_vis": A_vis,
            "R_bottleneck": R_r,
        }

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 0,
        return_attn_maps: bool = False,
        **kwargs,
    ) -> ReInspectionOutput:
        prepared = self._prepare_reinspection_inputs(
            input_ids, inputs_embeds, attention_mask, position_ids, labels,
            pixel_values, vision_feature_layer, vision_feature_select_strategy,
        )

        prepared["inputs_embeds"] = prepared["inputs_embeds"].to(self.base_model.dtype)

        outputs = self.base_model.model.language_model(
            position_ids=prepared["position_ids"],
            inputs_embeds=prepared["inputs_embeds"],
            attention_mask=prepared["attention_mask"],
            output_hidden_states=False,
            use_cache=False,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.base_model.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if prepared["labels"] is not None:
            labels = prepared["labels"]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = _chunked_cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        return ReInspectionOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=prepared["image_features"],
            attn_task=prepared["A_task"] if return_attn_maps else None,
            attn_vis=prepared["A_vis"] if return_attn_maps else None,
            R_bottleneck=prepared["R_bottleneck"] if return_attn_maps else None,
        )

    def get_attention_maps(self):
        """Return the last computed attention maps for visualization."""
        return self._last_attn_task, self._last_attn_vis

    @property
    def last_generation_prompt_lengths(self) -> Optional[torch.LongTensor]:
        return self._last_generation_prompt_lengths

    @torch.no_grad()
    def generate(self, **kwargs):
        input_ids = kwargs.pop("input_ids", None)
        pixel_values = kwargs.pop("pixel_values", None)
        attention_mask = kwargs.pop("attention_mask", None)
        vision_feature_layer = kwargs.pop("vision_feature_layer", None)
        vision_feature_select_strategy = kwargs.pop("vision_feature_select_strategy", None)

        if input_ids is None:
            raise ValueError("input_ids is required for generate()")

        prepared = self._prepare_reinspection_inputs(
            input_ids, None, attention_mask, None, None,
            pixel_values, vision_feature_layer, vision_feature_select_strategy,
        )
        self._last_generation_prompt_lengths = self._sequence_lengths(
            prepared["input_ids"], prepared["attention_mask"]
        ).detach().cpu()

        return self.base_model.generate(
            input_ids=prepared["input_ids"],
            inputs_embeds=prepared["inputs_embeds"],
            attention_mask=prepared["attention_mask"],
            **kwargs,
        )


def load_processor(config: ReInspectionConfig):
    return AutoProcessor.from_pretrained(config.processor_path)


def load_model(
    config: ReInspectionConfig,
    device_map: str = "auto",
    processor=None,
) -> InternVL3WithReInspection:
    """Load InternVL3-8B and wrap it with the re-inspection module."""
    base_model = InternVLForConditionalGeneration.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map=device_map,
    )
    return InternVL3WithReInspection(
        config=config,
        base_model=base_model,
        processor=processor,
    )
