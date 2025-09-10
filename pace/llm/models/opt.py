# *******************************************************************************
# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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

# Adapted from https://github.com/huggingface/transformers/blob/v4.48.2/src/transformers/models/opt/modeling_opt.py

from typing import Optional, List, Tuple, Iterable

import torch
from torch import nn
from transformers import OPTConfig
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from pace.llm.outputs import ModelOutput
from pace.llm.configs import OperatorConfig
from pace.llm.models.base_model import BaseModelForCausalLM
from pace.llm.cache import KVCacheBase, KVCacheManager
from pace.llm.ops import Linear, MultiHeadAttention, LayerNorm, MergedMLP


class OPTLearnedPositionalEmbedding(nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the
        # embedding ids by 2 and adjust num_embeddings appropriately. Other
        # models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(
        self, attention_mask: torch.Tensor, past_key_values_length: int = 0
    ) -> torch.Tensor:

        # create positions depending on attention_mask
        positions = (
            torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask
        ).long() - 1
        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
        opconfig: OperatorConfig,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.enable_bias = config.enable_bias
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        self.k_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            bias=self.enable_bias,
            backend_impl=opconfig.qkv_projection,
        )
        self.v_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            bias=self.enable_bias,
            backend_impl=opconfig.qkv_projection,
        )
        self.q_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            bias=self.enable_bias,
            backend_impl=opconfig.qkv_projection,
        )
        self.out_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            bias=self.enable_bias,
            backend_impl=opconfig.out_projection,
        )

        self.attention = MultiHeadAttention(backend_impl=opconfig.attention)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: KVCacheBase,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        # Make sure the hidden state dim is 3 and the third dim is same as embedding dim
        assert hidden_states.dim() == 3
        assert hidden_states.shape[2] == self.embed_dim

        batch_size, seq_len, _ = (
            hidden_states.shape
        )  # batch size, sequence length, embedding dim

        qkv_view_size = (batch_size, seq_len, self.num_heads, self.head_dim)

        key_states = (
            self.k_proj(hidden_states).view(*qkv_view_size).transpose(1, 2)
        )  # bs, num_head, seq_len, head_dim
        value_states = (
            self.v_proj(hidden_states).view(*qkv_view_size).transpose(1, 2)
        )  # bs, num_head, seq_len, head_dim
        query_states = (
            self.q_proj(hidden_states).view(*qkv_view_size).transpose(1, 2)
        )  # bs, num_head, seq_len, head_dim

        key_states, value_states = kv_cache.update_cache(
            key_states, value_states, concat_dim=2
        )

        attn_output = self.attention(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
        )  # bs, num_head, seq_len, head_dim

        attn_output = attn_output.transpose(
            1, 2
        ).contiguous()  # bs, seq_len, num_head, head_dim

        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)
        return attn_output


class OPTDecoderLayer(nn.Module):

    def __init__(self, config: OPTConfig, opconfig: OperatorConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size

        self.self_attn = OPTAttention(config=config, opconfig=opconfig)
        self.do_layer_norm_before = config.do_layer_norm_before

        self.self_attn_layer_norm = LayerNorm(
            self.embed_dim,
            elementwise_affine=config.layer_norm_elementwise_affine,
            backend_impl=opconfig.norm,
        )
        self.fc = MergedMLP(
            self.embed_dim,
            config.ffn_dim,
            bias=config.enable_bias,
            activation=config.activation_function,
            gate=False,
            backend_impl=opconfig.mlp,
        )
        self.final_layer_norm = LayerNorm(
            self.embed_dim,
            elementwise_affine=config.layer_norm_elementwise_affine,
            backend_impl=opconfig.norm,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: KVCacheBase,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        # Self Attention
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc(hidden_states)
        hidden_states = residual + hidden_states
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class OPTDecoder(nn.Module):

    def __init__(self, config: OPTConfig, opconfig: OperatorConfig):

        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.attn_mask_converter = AttentionMaskConverter(
            is_causal=True, sliding_window=None
        )

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.word_embed_proj_dim, self.padding_idx
        )
        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size
        )

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = Linear(
                config.hidden_size,
                config.word_embed_proj_dim,
                bias=False,
                backend_impl=(
                    opconfig["project_out"] if "project_out" in opconfig else None
                ),
            )
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = Linear(
                config.word_embed_proj_dim,
                config.hidden_size,
                bias=False,
                backend_impl=(
                    opconfig["project_in"] if "project_in" in opconfig else None
                ),
            )
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = LayerNorm(
                config.hidden_size,
                elementwise_affine=config.layer_norm_elementwise_affine,
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList(
            [OPTDecoderLayer(config, opconfig) for _ in range(config.num_hidden_layers)]
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

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
        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        # 4d mask is passed through the layers
        key_value_length = input_shape[-1] + past_key_values_length
        hf_attention_mask = self.attn_mask_converter.to_4d(
            attention_mask,
            input_shape[-1],
            key_value_length=key_value_length,
            dtype=inputs_embeds.dtype,
        )

        hf_attention_mask = kv_cache.update_mask(hf_attention_mask, org_input_size)

        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states, kv_cache.cache_objects[idx], hf_attention_mask
            )

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
        return hidden_states


class OPTModel(nn.Module):

    def __init__(self, config: OPTConfig, opconfig: OperatorConfig):
        super().__init__()
        self.decoder = OPTDecoder(config, opconfig)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.LongTensor,
        kv_cache: KVCacheManager,
        attention_mask: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        return self.decoder(
            input_ids, kv_cache, attention_mask, inputs_embeds=inputs_embeds
        )


class OPTForCausalLM(BaseModelForCausalLM):

    rename_layers = {
        "fc1": "fc.up_proj.linear",
        "fc2": "fc.down_proj",
    }

    def __init__(self, config: OPTConfig, opconfig: OperatorConfig):
        super().__init__(config)
        self.config = config

        self.model = OPTModel(config, opconfig)

        # Logits
        self.lm_head = Linear(
            config.word_embed_proj_dim,
            config.vocab_size,
            bias=False,
            backend_impl=opconfig.lm_head,
        )

    def load_weights(self, weight_iterator: Iterable[Tuple[str, torch.Tensor]]):

        params_dict = dict(self.named_parameters(remove_duplicate=False))

        found_lm_head = False
        for name, weight in weight_iterator:
            if name.startswith("decoder."):
                name = "model." + name
            name = self.rename_fused_params(name)

            if "lm_head.weight" in name:
                found_lm_head = True

            if name.endswith(".bias") and name not in params_dict:
                continue

            assert params_dict[name].size() == weight.size()
            params_dict[name].data.copy_(weight)

        if not found_lm_head:
            self.lm_head.weight = self.model.decoder.embed_tokens.weight

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
