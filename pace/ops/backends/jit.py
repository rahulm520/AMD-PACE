# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import Optional

import torch

import pace  # noqa: F401
from pace.ops.base import BackendBase
from pace.ops.registry import backend_registry
from pace.ops.enum import OperatorType, BackendType, DataType


@backend_registry.register(
    OperatorType.LINEAR, BackendType.JIT, [DataType.FLOAT32, DataType.BFLOAT16]
)
class JITLinear(BackendBase):

    def execute(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.ops.pace.linear(input, weight, bias)


@backend_registry.register(
    OperatorType.MHA, BackendType.JIT, [DataType.FLOAT32, DataType.BFLOAT16]
)
class JITAttention(BackendBase):

    def execute(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Ensure tensors are contiguous for JIT compatibility
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        return torch.ops.pace.attention(query, key, value, attention_mask, False)
