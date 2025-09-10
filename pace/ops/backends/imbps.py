# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import os
from typing import Optional

import torch
import torch.nn as nn

import pace  # noqa: F401
from pace.ops.base import BackendBase
from pace.ops.registry import backend_registry
from pace.ops.enum import FusedOperatorType, BackendType, DataType


@backend_registry.register(
    FusedOperatorType.FUSEDMLPLINEAR,
    BackendType.IMBPS,
    [DataType.FLOAT32, DataType.BFLOAT16],
)
class IMBPSLinear(BackendBase):

    def preprocess(self, layer):
        # Store chunked weights and biases in new attributes to avoid overwriting parameter attributes
        chunks = int(os.getenv("IMBPS_BLOCK_SIZE", 1))
        layer_up_proj_weight = layer.up_proj.linear.weight
        if layer.gate_proj:
            layer_gate_proj_weight = layer.gate_proj.linear.weight
            layer_gate_proj_bias = layer.gate_proj.linear.bias
        else:
            None
        layer_up_proj_bias = layer.up_proj.linear.bias

        layer.up_proj_weight_chunks = [
            w.contiguous() for w in layer_up_proj_weight.chunk(chunks=chunks)
        ]
        layer.down_proj_weight_chunks = [
            w.contiguous() for w in layer.down_proj.weight.chunk(chunks=chunks, dim=1)
        ]

        layer.gate_proj_weight_chunks = (
            [w.contiguous() for w in layer_gate_proj_weight.chunk(chunks=chunks)]
            if layer.gate_proj
            else None
        )
        if layer_up_proj_bias is not None:
            layer_up_proj_bias = nn.Parameter(layer_up_proj_bias)
        if layer.down_proj.bias is not None:
            layer.down_proj.bias = nn.Parameter(layer.down_proj.bias)
        if layer_up_proj_bias is not None:
            layer.up_proj_bias_chunks = [
                b.contiguous() for b in layer_up_proj_bias.chunk(chunks=chunks)
            ]
        else:
            layer.up_proj_bias_chunks = None

        layer.gate_proj_bias_chunks = (
            [b.contiguous() for b in layer_gate_proj_bias.chunk(chunks=chunks)]
            if layer.gate_proj and layer_gate_proj_bias is not None
            else None
        )

    def execute(
        self,
        input: torch.Tensor,
        up_proj_weights: torch.Tensor,
        up_proj_bias: Optional[torch.Tensor],
        down_proj_weights: torch.Tensor,
        down_proj_bias: Optional[torch.Tensor],
        activation: Optional[str],
        gate_proj_weights: Optional[torch.Tensor],
        gate_proj_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:

        original_shape = input.shape
        return (
            torch.ops.pace.mlp_mlp_fusion(
                input,
                up_proj_weights,
                up_proj_bias,
                down_proj_weights,
                down_proj_bias,
                activation,
                gate_proj_weights,
                gate_proj_bias,
            )
            .reshape(original_shape)
            .contiguous()
        )
