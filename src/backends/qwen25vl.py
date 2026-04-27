"""Qwen2.5-VL with Re-Inspection Module.

Wraps Qwen2_5_VLForConditionalGeneration, inserting task-conditioned
re-inspection tokens (R) between the user message and assistant response.

Uses d_model=3584, MRoPE [16,24,24], and the <|im_start|>/<|im_end|> chat
template. Same wrapping pattern as the InternVL3/Gemma4 backends.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from src.backends.hf_hub_utils import resolve_pretrained_local_path
from src.config import ReInspectionConfig
from src.model.outputs import ReInspectionOutput
from src.model.reinspection_module import ReInspectionModule


class Qwen25VLWithReInspection(nn.Module):
    """Qwen2.5-VL + Re-Inspection Module.

    Wrapper: embed, encode vision, scatter, inject R tokens before
    <|im_start|>assistant, forward through LLM.
    """

    def __init__(self, config: ReInspectionConfig, base_model: Qwen2_5_VLForConditionalGeneration):
        super().__init__()
        self.config = config
        self.base_model = base_model
        target_dtype = torch.bfloat16 if config.bf16 else torch.float32
        self.reinspection = ReInspectionModule(config, dtype=target_dtype)

        self._im_start_id = base_model.config.im_start_id if hasattr(base_model.config, 'im_start_id') else None
        self._image_token_id = base_model.config.image_token_id
        self._video_token_id = base_model.config.video_token_id

        self._last_attn_task = None
        self._last_attn_vis = None

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    @property
    def dtype(self):
        return next(self.base_model.parameters()).dtype

    def _find_assistant_start_positions(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Find the position of the last <|im_start|> token in each sequence."""
        B, L = input_ids.shape

        if self._im_start_id is not None:
            mask = (input_ids == self._im_start_id)
            has_match = mask.any(dim=1)
            indices = mask.long() * torch.arange(L, device=input_ids.device).unsqueeze(0)
            positions = indices.max(dim=1).values
            positions[~has_match] = L
        else:
            positions = torch.full((B,), L, dtype=torch.long, device=input_ids.device)

        return positions

    def _compute_mrope_position_ids(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor],
        video_grid_thw: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Compute 3D MRoPE position IDs for Qwen2.5-VL.

        MRoPE scheme: [16, 24, 24] section split.
        """
        model = self.base_model.model
        cfg = model.config
        spatial_merge_size = cfg.vision_config.spatial_merge_size
        image_token_id = cfg.image_token_id
        video_token_id = cfg.video_token_id
        vision_start_token_id = cfg.vision_start_token_id

        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw = video_grid_thw.clone()
            video_grid_thw[:, 0] = 1

        B, L = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.ones(3, B, L, dtype=torch.long, device=input_ids.device)
        mrope_position_deltas = []
        image_index, video_index = 0, 0

        for i in range(B):
            seq = input_ids[i][attention_mask[i] == 1]
            input_tokens = seq.tolist()

            vision_start_indices = (seq == vision_start_token_id).nonzero(as_tuple=True)[0]
            vision_tokens = seq[vision_start_indices + 1] if len(vision_start_indices) else seq.new_empty(0)
            image_nums = int((vision_tokens == image_token_id).sum())
            video_nums = int((vision_tokens == video_token_id).sum())

            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            for _ in range(image_nums + video_nums):
                ed_image = input_tokens.index(image_token_id, st) if image_token_id in input_tokens[st:] and remain_images > 0 else len(input_tokens) + 1
                ed_video = input_tokens.index(video_token_id, st) if video_token_id in input_tokens[st:] and remain_videos > 0 else len(input_tokens) + 1

                if ed_image < ed_video:
                    t = image_grid_thw[image_index][0].item()
                    h = image_grid_thw[image_index][1].item() // spatial_merge_size
                    w = image_grid_thw[image_index][2].item() // spatial_merge_size
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t = video_grid_thw[video_index][0].item()
                    h = video_grid_thw[video_index][1].item() // spatial_merge_size
                    w = video_grid_thw[video_index][2].item() // spatial_merge_size
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                text_len = ed - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                if text_len > 0:
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                    st_idx = llm_pos_ids_list[-1].max() + 1

                t_idx = torch.arange(t).view(-1, 1).expand(-1, h * w).flatten()
                h_idx = torch.arange(h).view(1, -1, 1).expand(t, -1, w).flatten()
                w_idx = torch.arange(w).view(1, 1, -1).expand(t, h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_idx, h_idx, w_idx]) + st_idx)
                st = ed + t * h * w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(input_tokens))

        rope_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, rope_deltas

    def _extract_vision_and_text(
        self,
        inputs_embeds: torch.FloatTensor,
        input_ids: torch.LongTensor,
        insert_positions: torch.LongTensor,
        vision_mask: torch.BoolTensor = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
        """Extract vision tokens V and text tokens T from mixed inputs_embeds."""
        B, L, D = inputs_embeds.shape

        if vision_mask is None:
            vision_mask = (input_ids == self._image_token_id) | (input_ids == self._video_token_id)

        V_list = []
        T_list = []
        for b in range(B):
            vis_positions = vision_mask[b].nonzero(as_tuple=False).squeeze(-1)
            insert_pos = insert_positions[b].item()

            text_mask_b = ~vision_mask[b].clone()
            text_mask_b[insert_pos:] = False
            text_positions = text_mask_b.nonzero(as_tuple=False).squeeze(-1)

            V_list.append(inputs_embeds[b, vis_positions])
            T_list.append(inputs_embeds[b, text_positions])

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
            V_mask[b, :nv] = True
            T_mask[b, :nt] = True

        return V, T, vision_mask, V_mask, T_mask

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
        """Insert R tokens into the sequence at insert_positions.

        ``torch.cat`` preserves gradients through ``R`` for LM loss; backbone
        segments are detached.
        """
        B, L, D = inputs_embeds.shape
        N_q = R.shape[1]
        new_L = L + N_q

        embed_rows = []
        for b in range(B):
            pos = insert_positions[b].item()
            embed_rows.append(
                torch.cat(
                    [
                        inputs_embeds[b, :pos].detach(),
                        R[b],
                        inputs_embeds[b, pos:].detach(),
                    ],
                    dim=0,
                )
            )
        new_embeds = torch.stack(embed_rows, dim=0)

        new_attention_mask = None
        if attention_mask is not None:
            new_attention_mask = torch.zeros(B, new_L, device=attention_mask.device, dtype=attention_mask.dtype)

        new_labels = None
        if labels is not None:
            new_labels = torch.full((B, new_L), -100, device=labels.device, dtype=labels.dtype)

        new_input_ids = None
        if input_ids is not None:
            new_input_ids = torch.zeros(B, new_L, device=input_ids.device, dtype=input_ids.dtype)

        for b in range(B):
            pos = insert_positions[b].item()

            if attention_mask is not None:
                new_attention_mask[b, :pos] = attention_mask[b, :pos]
                new_attention_mask[b, pos:pos + N_q] = 1
                new_attention_mask[b, pos + N_q:] = attention_mask[b, pos:]

            if labels is not None:
                new_labels[b, :pos] = labels[b, :pos]
                new_labels[b, pos + N_q:] = labels[b, pos:]

            if input_ids is not None:
                new_input_ids[b, :pos] = input_ids[b, :pos]
                new_input_ids[b, pos + N_q:] = input_ids[b, pos:]

        new_position_ids = None
        if position_ids is not None:
            ndim = position_ids.ndim
            if ndim == 3:
                n_dims = position_ids.shape[0]
                new_position_ids = torch.zeros(
                    n_dims, B, new_L,
                    device=position_ids.device, dtype=position_ids.dtype,
                )
                for b in range(B):
                    pos = insert_positions[b].item()
                    for d in range(n_dims):
                        new_position_ids[d, b, :pos] = position_ids[d, b, :pos]

                        if pos > 0:
                            last_pos = position_ids[d, b, pos - 1].item()
                        else:
                            last_pos = -1

                        r_ids = torch.arange(N_q, device=position_ids.device) + last_pos + 1
                        new_position_ids[d, b, pos:pos + N_q] = r_ids

                        after_ids = position_ids[d, b, pos:] + N_q
                        new_position_ids[d, b, pos + N_q:] = after_ids
            else:
                new_position_ids = torch.zeros(B, new_L, device=position_ids.device, dtype=position_ids.dtype)
                for b in range(B):
                    pos = insert_positions[b].item()
                    new_position_ids[b, :pos] = position_ids[b, :pos]
                    last_pos = position_ids[b, pos - 1].item() if pos > 0 else -1
                    new_position_ids[b, pos:pos + N_q] = torch.arange(N_q, device=position_ids.device) + last_pos + 1
                    new_position_ids[b, pos + N_q:] = position_ids[b, pos:] + N_q

        return {
            "inputs_embeds": new_embeds,
            "attention_mask": new_attention_mask,
            "position_ids": new_position_ids,
            "labels": new_labels,
            "input_ids": new_input_ids,
        }

    def _encode_vision_and_scatter(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """Encode vision features and scatter into inputs_embeds.

        Qwen2.5-VL does not use DeepStack, so we return only the updated embeds.
        """
        model = self.base_model.model

        if pixel_values is not None:
            pixel_values = pixel_values.type(model.visual.dtype)
            image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = (input_ids == self._image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
            video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            video_mask = (input_ids == self._video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        return inputs_embeds

    def _prepare_reinspection_inputs(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        labels: Optional[torch.LongTensor],
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        need_weights: bool = False,
    ) -> dict:
        """Prepare inputs with vision encoding, reinspection, and token insertion."""
        model = self.base_model.model

        if inputs_embeds is None:
            inputs_embeds = model.get_input_embeddings()(input_ids)

        inputs_embeds = self._encode_vision_and_scatter(
            input_ids, inputs_embeds, pixel_values, pixel_values_videos,
            image_grid_thw, video_grid_thw,
        )

        if position_ids is None:
            position_ids, rope_deltas = self._compute_mrope_position_ids(
                input_ids, image_grid_thw, video_grid_thw, attention_mask,
            )
            model.rope_deltas = rope_deltas

        insert_positions = self._find_assistant_start_positions(input_ids)
        V, T, _, V_mask, T_mask = self._extract_vision_and_text(inputs_embeds, input_ids, insert_positions)
        R, A_task, A_vis, R_r, Q_task, T_down = self.reinspection(
            V, T, V_mask=V_mask, T_mask=T_mask, need_weights=need_weights,
        )

        if need_weights:
            self._last_attn_task = A_task.detach().cpu()
            self._last_attn_vis = A_vis.detach().cpu()
        else:
            self._last_attn_task = None
            self._last_attn_vis = None

        inserted = self._insert_tokens(
            inputs_embeds, R, insert_positions,
            attention_mask, position_ids, labels, input_ids,
        )

        return {
            **inserted,
            "A_task": A_task,
            "A_vis": A_vis,
            "R_bottleneck": R_r,
            "Q_task": Q_task,
            "T_down": T_down,
            "T_mask": T_mask,
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
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 0,
        return_attn_maps: bool = False,
        return_query_text_tensors: bool = False,
        **kwargs,
    ) -> ReInspectionOutput:
        """Forward pass with Re-Inspection token injection."""
        prepared = self._prepare_reinspection_inputs(
            input_ids, inputs_embeds, attention_mask, position_ids, labels,
            pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw,
            need_weights=return_attn_maps,
        )

        outputs = self.base_model.model.language_model(
            input_ids=None,
            position_ids=prepared["position_ids"],
            attention_mask=prepared["attention_mask"],
            past_key_values=past_key_values,
            inputs_embeds=prepared["inputs_embeds"],
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        if logits_to_keep > 0:
            logits = self.base_model.lm_head(hidden_states[:, -logits_to_keep:, :])
        else:
            logits = self.base_model.lm_head(hidden_states)

        loss = None
        if prepared["labels"] is not None:
            loss = self.base_model.loss_function(
                logits=logits,
                labels=prepared["labels"],
                vocab_size=self.base_model.config.text_config.vocab_size,
            )

        return ReInspectionOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=getattr(self.base_model.model, 'rope_deltas', None),
            attn_task=prepared["A_task"] if return_attn_maps else None,
            attn_vis=prepared["A_vis"] if return_attn_maps else None,
            R_bottleneck=prepared["R_bottleneck"] if return_attn_maps else None,
            Q_text_bottleneck=prepared["Q_task"] if return_query_text_tensors else None,
            text_bottleneck=prepared["T_down"] if return_query_text_tensors else None,
            text_bottleneck_mask=prepared["T_mask"] if return_query_text_tensors else None,
        )

    def get_attention_maps(self):
        """Return the last computed attention maps for visualization."""
        return self._last_attn_task, self._last_attn_vis

    @torch.no_grad()
    def generate(self, **kwargs):
        """Generation with R token injection via forward hook.

        Strategy: insert N_q placeholder tokens, let the base model handle
        vision encoding and MRoPE, then replace placeholders with actual R
        tokens via a pre-hook on language_model.
        """
        input_ids = kwargs.pop("input_ids", None)
        pixel_values = kwargs.pop("pixel_values", None)
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        attention_mask = kwargs.pop("attention_mask", None)

        if input_ids is None:
            raise ValueError("input_ids is required for generate()")

        N_q = self.config.n_queries
        B, L = input_ids.shape
        device = input_ids.device

        insert_positions = self._find_assistant_start_positions(input_ids)

        pad_id = getattr(self.base_model.config, 'pad_token_id', 0) or 0
        new_L = L + N_q
        new_input_ids = torch.full((B, new_L), pad_id, device=device, dtype=input_ids.dtype)
        new_attention_mask = (
            torch.ones(B, new_L, device=device, dtype=attention_mask.dtype)
            if attention_mask is not None else None
        )

        for b in range(B):
            pos = insert_positions[b].item()
            new_input_ids[b, :pos] = input_ids[b, :pos]
            new_input_ids[b, pos + N_q:] = input_ids[b, pos:]

            if attention_mask is not None:
                new_attention_mask[b, :pos] = attention_mask[b, :pos]
                new_attention_mask[b, pos:pos + N_q] = 1
                new_attention_mask[b, pos + N_q:] = attention_mask[b, pos:]

        is_prefill = [True]

        def _inject_reinspection(module, args, hook_kwargs):
            if not is_prefill[0]:
                return
            is_prefill[0] = False

            inputs_embeds = hook_kwargs['inputs_embeds']

            vision_mask = None
            if input_ids is not None:
                orig_vis_mask = (input_ids == self._image_token_id) | (input_ids == self._video_token_id)
                vision_mask = torch.zeros(B, new_L, dtype=torch.bool, device=device)
                for b_idx in range(B):
                    p = insert_positions[b_idx].item()
                    vision_mask[b_idx, :p] = orig_vis_mask[b_idx, :p]
                    vision_mask[b_idx, p + N_q:] = orig_vis_mask[b_idx, p:]

            V, T, _, V_mask, T_mask = self._extract_vision_and_text(
                inputs_embeds, None, insert_positions, vision_mask=vision_mask,
            )

            R, A_task, A_vis, _R_r, _Q_task, _T_down = self.reinspection(
                V, T, V_mask=V_mask, T_mask=T_mask, need_weights=True,
            )
            self._last_attn_task = A_task.detach().cpu() if A_task is not None else None
            self._last_attn_vis = A_vis.detach().cpu() if A_vis is not None else None

            new_embeds = inputs_embeds.clone()
            for b_idx in range(B):
                pos = insert_positions[b_idx].item()
                new_embeds[b_idx, pos:pos + N_q] = R[b_idx]
            hook_kwargs['inputs_embeds'] = new_embeds

            return args, hook_kwargs

        handle = self.base_model.model.language_model.register_forward_pre_hook(
            _inject_reinspection, with_kwargs=True,
        )

        try:
            gen_kwargs = dict(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                **kwargs,
            )
            return self.base_model.generate(**gen_kwargs)
        finally:
            handle.remove()


def load_model(config: ReInspectionConfig, device_map: str = "auto") -> Qwen25VLWithReInspection:
    """Load Qwen2.5-VL and wrap with Re-Inspection Module."""
    resolved = resolve_pretrained_local_path(config.model_name_or_path)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        resolved,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map=device_map,
    )
    return Qwen25VLWithReInspection(config, base_model)
