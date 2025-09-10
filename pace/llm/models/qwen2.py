# *******************************************************************************
# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/huggingface/transformers/blob/v4.48.2/src/transformers/models/qwen2/modeling_qwen2.py

from typing import Optional, Tuple, Iterable, Callable

import torch
from torch import nn
from transformers import Qwen2Config
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from pace.llm.outputs import ModelOutput
from pace.llm.configs import OperatorConfig
from pace.llm.models.base_model import BaseModelForCausalLM
from pace.llm.cache import KVCacheBase, KVCacheManager
from pace.llm.ops import (
    Linear,
    # RepeatedKVLinear,
    MultiHeadAttention,
    RMSNorm,
    RotaryEmbedding,
    MergedMLP,
)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2Attention(nn.Module):

    def __init__(self, config: Qwen2Config, opconfig: OperatorConfig):
        super().__init__()

        self.config = config
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.q_proj = Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=True,
            backend_impl=opconfig.qkv_projection,
        )
        self.k_proj = Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=True,
            backend_impl=opconfig.qkv_projection,
        )
        self.v_proj = Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=True,
            backend_impl=opconfig.qkv_projection,
        )
        self.attention = MultiHeadAttention(backend_impl=opconfig.attention)
        self.o_proj = Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            backend_impl=opconfig.out_projection,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Callable,
        kv_cache: KVCacheBase,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        ...

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states, key_states = position_embeddings(query_states, key_states)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # KV Cache
        key_states, value_states = kv_cache.update_cache(
            key_states, value_states, concat_dim=2
        )

        attn_output = self.attention(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output


class Qwen2DecoderLayer(nn.Module):

    def __init__(self, config: Qwen2Config, opconfig: OperatorConfig):
        super().__init__()
        self.config = config
        self.self_attn = Qwen2Attention(config, opconfig)
        self.mlp = MergedMLP(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            activation=config.hidden_act,
            gate=True,
            backend_impl=opconfig.mlp,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, backend_impl=opconfig.norm
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, backend_impl=opconfig.norm
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Callable,
        kv_cache: KVCacheBase,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states, position_embeddings, kv_cache, attention_mask
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen2Model(nn.Module):

    def __init__(self, config: Qwen2Config, opconfig: OperatorConfig):
        super().__init__()

        self.config = config
        self.padding_idx = config.pad_token_id

        self.attn_mask_converter = AttentionMaskConverter(
            is_causal=True, sliding_window=None
        )

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(config, opconfig)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, backend_impl=opconfig.norm
        )
        self.rotary_emb = RotaryEmbedding(
            rope_scaling=config.rope_scaling,
            rotary_dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            backend_impl=opconfig.rope,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        kv_cache: KVCacheManager,
        attention_mask: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )

        past_key_values_length = len(kv_cache)

        org_input_size = input_ids.size(1)
        input_ids = input_ids[:, past_key_values_length:]

        input_shape = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # 4d mask is passed through the layers
        key_value_length = input_shape[-1] + past_key_values_length
        hf_attention_mask = self.attn_mask_converter.to_4d(
            attention_mask,
            input_shape[-1],
            key_value_length=key_value_length,
            dtype=inputs_embeds.dtype,
        )
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if kv_cache:
            position_ids = position_ids[:, -input_ids.shape[1] :]

        hidden_states = inputs_embeds
        position_embeddings: Callable = self.rotary_emb(hidden_states, position_ids)

        hf_attention_mask = kv_cache.update_mask(hf_attention_mask, org_input_size)

        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states, position_embeddings, kv_cache[idx], hf_attention_mask
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen2ForCausalLM(BaseModelForCausalLM):

    rename_layers = {
        "up_proj": "up_proj.linear",
        "gate_proj": "gate_proj.linear",
    }

    def __init__(self, config: Qwen2Config, opconfig: OperatorConfig):
        super().__init__(config)
        self.config = config
        self.model = Qwen2Model(config, opconfig)
        self.lm_head = Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            backend_impl=opconfig.lm_head,
        )

    def load_weights(self, weight_iterator: Iterable[Tuple[str, torch.Tensor]]):

        params_dict = dict(self.named_parameters(remove_duplicate=False))

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        for name, weight in weight_iterator:
            name = self.rename_fused_params(name)
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue

            assert params_dict[name].size() == weight.size()
            params_dict[name].data.copy_(weight)

    def forward(
        self,
        input_ids: torch.LongTensor,
        kv_cache: KVCacheManager,
        attention_mask: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        model_output = self.model(
            input_ids, kv_cache, attention_mask, inputs_embeds=inputs_embeds
        )
        logits = self.lm_head(model_output)

        return ModelOutput(logits=logits)
