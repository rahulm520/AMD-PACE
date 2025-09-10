# *******************************************************************************
# Modifications Copyright (c) 2024 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************
# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
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

# Adapted from https://github.com/huggingface/transformers/blob/v4.48.2/src/transformers/models/gptj/modeling_gptj.py

from typing import Optional, Tuple, Iterable

import torch
from torch import nn
from transformers import GPTJConfig
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from pace.llm.outputs import ModelOutput
from pace.llm.configs import OperatorConfig
from pace.llm.cache import KVCacheBase, KVCacheManager
from pace.llm.models.base_model import BaseModelForCausalLM
from pace.llm.ops import (
    Linear,
    MultiHeadAttention,
    LayerNorm,
    RotaryEmbedding,
    MergedMLP,
)


class GPTJAttention(nn.Module):

    def __init__(self, config: GPTJConfig, opconfig: OperatorConfig):

        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads

        self.k_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            bias=False,
            backend_impl=opconfig.qkv_projection,
        )
        self.v_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            bias=False,
            backend_impl=opconfig.qkv_projection,
        )
        self.q_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            bias=False,
            backend_impl=opconfig.qkv_projection,
        )

        self.rotary_dim = config.rotary_dim
        pos_embd_dim = self.rotary_dim or self.embed_dim
        self.rotary_emb = RotaryEmbedding(
            rope_scaling=(
                config.rope_scaling if hasattr(config, "rope_scaling") else None
            ),
            rotary_dim=pos_embd_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=getattr(config, "rope_theta", 10000),
            backend_impl=opconfig.rope,
        )
        self.attention = MultiHeadAttention(backend_impl=opconfig.attention)
        self.out_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            bias=False,
            backend_impl=opconfig.out_projection,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Tuple[torch.Tensor, torch.Tensor],
        kv_cache: KVCacheBase,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        if self.rotary_dim is not None:
            key_rot = key_states[..., : self.rotary_dim]
            key_pass = key_states[..., self.rotary_dim :]

            query_rot = query_states[..., : self.rotary_dim]
            query_pass = query_states[..., self.rotary_dim :]

            query_rot, key_rot = position_embeddings(
                query_rot, key_rot, is_neox_style=False
            )
            key_states = torch.cat([key_rot, key_pass], dim=-1)
            query_states = torch.cat([query_rot, query_pass], dim=-1)
        else:
            query_states, key_states = position_embeddings(
                query_states, key_states, is_neox_style=False
            )

        # KV Cache
        if kv_cache is not None:
            key_states, value_states = kv_cache.update_cache(
                key_states, value_states, concat_dim=2
            )

        attn_output = self.attention(
            query_states, key_states, value_states, attention_mask=attention_mask
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.out_proj(attn_output)

        return attn_output


class GPTJBlock(nn.Module):

    def __init__(
        self,
        config: GPTJConfig,
        opconfig: OperatorConfig,
    ):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = LayerNorm(
            config.n_embd, eps=config.layer_norm_epsilon, backend_impl=opconfig.norm
        )
        self.attn = GPTJAttention(config, opconfig)
        self.mlp = MergedMLP(
            config.n_embd,
            inner_dim,
            bias=True,
            activation=config.activation_function,
            backend_impl=opconfig.mlp,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Tuple[torch.Tensor, torch.Tensor],
        kv_cache: KVCacheBase,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, position_ids, kv_cache, attention_mask)

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

        return hidden_states


class GPTJModel(nn.Module):

    def __init__(self, config: GPTJConfig, opconfig: OperatorConfig):

        super().__init__()

        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(self.vocab_size, self.embed_dim)
        self.h = nn.ModuleList(
            [GPTJBlock(config, opconfig) for _ in range(config.n_layer)]
        )
        self.ln_f = LayerNorm(
            self.embed_dim, eps=config.layer_norm_epsilon, backend_impl=opconfig.norm
        )

        self.attn_mask_converter = AttentionMaskConverter(
            is_causal=True, sliding_window=None
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        kv_cache: KVCacheManager,
        attention_mask: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )

        past_key_values_length = len(kv_cache)
        org_input_size = input_ids.size(1)
        input_ids = input_ids[:, past_key_values_length:]

        input_shape = input_ids.shape
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

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

        hf_attention_mask = kv_cache.update_mask(hf_attention_mask, org_input_size)
        for idx, decoder_layer in enumerate(self.h):
            hidden_states = decoder_layer(
                hidden_states, position_ids, kv_cache[idx], hf_attention_mask
            )

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class GPTJForCausalLM(BaseModelForCausalLM):

    rename_layers = {
        "fc_in": "up_proj.linear",
        "fc_out": "down_proj",
    }

    def __init__(self, config: GPTJConfig, opconfig: OperatorConfig):
        super().__init__(config)
        self.config = config

        self.transformer = GPTJModel(config, opconfig)
        self.lm_head = Linear(
            config.n_embd, config.vocab_size, backend_impl=opconfig.lm_head
        )

    def load_weights(self, weight_iterator: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, weight in weight_iterator:
            name = self.rename_fused_params(name)

            if "attn.bias" in name or "attn.masked_bias" in name:
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
        model_output = self.transformer(
            input_ids, kv_cache, attention_mask, inputs_embeds=inputs_embeds
        )
        logits = self.lm_head(model_output)

        return ModelOutput(logits=logits)
