# *******************************************************************************
# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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

# Adapted from https://github.com/huggingface/transformers/blob/v4.48.2/src/transformers/models/llama/modeling_llama.py
# The file contains the implemention of LLAMA models, as well as Phi3/4 models. Since they are the same,
# the implementation is shared between the two models.

from typing import Optional, List, Tuple, Iterable, Union, Callable

import torch
from torch import nn
from transformers import LlamaConfig, PhiConfig
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from pace.llm.outputs import ModelOutput
from pace.llm.configs import OperatorConfig
from pace.llm.cache import KVCacheBase, KVCacheManager
from pace.llm.models.base_model import BaseModelForCausalLM
from pace.llm.ops import (
    Linear,
    # RepeatedKVLinear,
    MultiHeadAttention,
    RMSNorm,
    RotaryEmbedding,
    MergedMLP,
)


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


class LlamaAttention(nn.Module):
    def __init__(self, config: Union[LlamaConfig, PhiConfig], opconfig: OperatorConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        phi_model = config.architectures[0] == "Phi3ForCausalLM"
        bias = False if phi_model else config.attention_bias

        self.q_proj = Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=bias,
            backend_impl=opconfig.qkv_projection,
        )
        self.k_proj = Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=bias,
            backend_impl=opconfig.qkv_projection,
        )
        self.v_proj = Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=bias,
            backend_impl=opconfig.qkv_projection,
        )
        # self.k_proj = RepeatedKVLinear(
        #     self.hidden_size, self.num_heads * self.head_dim, bias=bias, num_key_value_heads=self.num_key_value_heads, backend_impl=opconfig.qkv_projection
        # )
        # self.v_proj = RepeatedKVLinear(
        #     self.hidden_size, self.num_heads * self.head_dim, bias=bias, num_key_value_heads=self.num_key_value_heads, backend_impl=opconfig.qkv_projection
        # )
        self.attention = MultiHeadAttention(backend_impl=opconfig.attention)
        self.o_proj = Linear(
            self.hidden_size,
            self.hidden_size,
            bias=config.attention_bias,
            backend_impl=opconfig.out_projection,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Callable,
        kv_cache: KVCacheBase,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Make sure the hidden state dim is 3 and the third dim is same as embedding dim
        assert hidden_states.dim() == 3
        assert hidden_states.shape[2] == self.hidden_size

        bsz, q_len, _ = hidden_states.size()

        # Do QKV linear projection
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape the QKV states
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        # key_states = key_states.view(
        #     bsz, q_len, self.num_heads, self.head_dim
        # ).transpose(1, 2)
        # value_states = value_states.view(
        #     bsz, q_len, self.num_heads, self.head_dim
        # ).transpose(1, 2)

        # Apply rotary embedding
        query_states, key_states = position_embeddings(
            query=query_states, key=key_states
        )

        # Convert GQA to MHA
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
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output


class LlamaDecoderLayer(nn.Module):

    def __init__(self, config: Union[LlamaConfig, PhiConfig], opconfig: OperatorConfig):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend_impl=opconfig.norm,
        )
        self.self_attn = LlamaAttention(config, opconfig)
        self.mlp = MergedMLP(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            bias=(
                False
                if config.architectures[0] == "Phi3ForCausalLM"
                else config.mlp_bias
            ),
            activation=config.hidden_act,
            gate=True,
            backend_impl=opconfig.mlp,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend_impl=opconfig.norm,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Callable,
        kv_cache: KVCacheBase,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        # Attn
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, position_embeddings, kv_cache, attention_mask
        )
        hidden_states = hidden_states + residual

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class LlamaModel(nn.Module):

    def __init__(self, config: Union[LlamaConfig, PhiConfig], opconfig: OperatorConfig):
        super().__init__()

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.attn_mask_converter = AttentionMaskConverter(
            is_causal=True, sliding_window=None
        )

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, opconfig)
                for _ in range(config.num_hidden_layers)
            ]
        )

        # Phi models introduced a partial_rotary_factor parameter in the config
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)
        self.rotary_emb = RotaryEmbedding(
            rope_scaling=config.rope_scaling,
            rotary_dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            backend_impl=opconfig.rope,
            original_max_position_embeddings=getattr(
                config, "original_max_position_embeddings", None
            ),
        )
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend_impl=opconfig.norm,
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
        # RotaryEmbedding forward will return a callable that can be used
        # to apply the rotary embedding to the query and key states
        position_embeddings: Callable = self.rotary_emb(hidden_states, position_ids)

        hf_attention_mask = kv_cache.update_mask(hf_attention_mask, org_input_size)

        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states, position_embeddings, kv_cache[idx], hf_attention_mask
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCausalLM(BaseModelForCausalLM):

    target_map = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    rename_layers = {
        "up_proj": "up_proj.linear",
        "gate_proj": "gate_proj.linear",
    }

    def __init__(self, config: Union[LlamaConfig, PhiConfig], opconfig: OperatorConfig):
        super().__init__(config)
        self.config = config
        self.model = LlamaModel(config, opconfig)
        self.vocab_size = config.vocab_size
        self.lm_head = Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            backend_impl=opconfig.lm_head,
        )

    def load_weights(self, weight_iterator: Iterable[Tuple[str, torch.Tensor]]):

        def split_projection_weight_path(input_string, target_map):
            import re

            """
            Splits a combined projection weight path into separate paths based on a target map.

            Args:
                input_string: The input string representing the combined weight path.
                target_map: A dictionary where keys are target names (e.g., "qkv_proj", "gate_up_proj")
                        and values are lists of corresponding split names (e.g., ["q_proj", "k_proj", "v_proj"], ["gate_proj", "up_proj"]).

            Returns:
                A tuple containing the split weight paths, or None if the
                input string doesn't match the expected pattern or the target_name is not in target_map.
                Returns an empty tuple if the target name already exists.
            """
            for target_name, split_names in target_map.items():
                match = re.search(rf"(.*{target_name})(.*)", input_string)
                if match:
                    prefix = match.group(1)[: -len(target_name)]
                    suffix = match.group(2)

                    split_paths = [f"{prefix}{name}{suffix}" for name in split_names]
                    return tuple(split_paths)
            return None

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

            # For Phi models, break the qkv projection into 3 parts according
            # to num_kv_heads and num_attention_heads
            # and gate_up_proj into 2 parts
            if self.target_map:
                split_names = split_projection_weight_path(name, self.target_map)
                if split_names is not None:
                    if name.endswith("qkv_proj.weight") and hasattr(
                        self.config, "num_key_value_heads"
                    ):
                        head_dim = (
                            self.config.hidden_size // self.config.num_attention_heads
                        )
                        split_weights = weight.split(
                            [
                                self.config.num_attention_heads * head_dim,
                                self.config.num_key_value_heads * head_dim,
                                self.config.num_key_value_heads * head_dim,
                            ]
                        )
                    else:
                        split_weights = weight.chunk(len(split_names), dim=0)

                    for split_name, split_weight in zip(split_names, split_weights):
                        assert params_dict[split_name].size() == split_weight.size()
                        params_dict[split_name].data.copy_(split_weight)
                    continue

            if hasattr(params_dict[name], "load_weights"):
                # If the parameter is a Linear or RepeatedKVLinear, use its load_weights method
                params_dict[name].load_weights(params_dict[name], weight)
                continue

            # Otherwise, copy the weight directly
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
