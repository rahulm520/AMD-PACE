# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from pace.ops.base import BackendBase
from pace.ops.registry import backend_registry
from pace.ops.enum import OperatorType, BackendType, DataType


@backend_registry.register(
    OperatorType.LINEAR, BackendType.NATIVE, [DataType.FLOAT32, DataType.BFLOAT16]
)
class NativeLinear(BackendBase):

    def execute(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.linear(input, weight, bias)


@backend_registry.register(
    OperatorType.MHA, BackendType.NATIVE, [DataType.FLOAT32, DataType.BFLOAT16]
)
class NativeAttention(BackendBase):

    def execute(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
            return F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask
            )


@backend_registry.register(
    OperatorType.LAYERNORM, BackendType.NATIVE, [DataType.FLOAT32, DataType.BFLOAT16]
)
class NativeLayerNorm(BackendBase):

    def execute(
        self,
        input: torch.Tensor,
        normalized_shape: Tuple[int],
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        return F.layer_norm(input, normalized_shape, weight, bias, eps)


@backend_registry.register(
    OperatorType.RMSNORM, BackendType.NATIVE, [DataType.FLOAT32, DataType.BFLOAT16]
)
class NativeRMSNorm(BackendBase):

    def execute(
        self,
        input: torch.Tensor,
        normalized_shape: Tuple[int],
        weight: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
    ) -> torch.Tensor:

        return F.rms_norm(
            input, normalized_shape=normalized_shape, weight=weight, eps=eps
        )


@backend_registry.register(
    OperatorType.SOFTMAX, BackendType.NATIVE, [DataType.FLOAT32, DataType.BFLOAT16]
)
class NativeSoftmax(BackendBase):

    def execute(
        self,
        input: torch.Tensor,
        dim: int = -1,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        return F.softmax(input, dim=dim, dtype=dtype)


@backend_registry.register(
    OperatorType.RELU, BackendType.NATIVE, [DataType.FLOAT32, DataType.BFLOAT16]
)
class NativeReLU(BackendBase):

    def execute(
        self,
        input: torch.Tensor,
        inplace: bool = False,
    ) -> torch.Tensor:
        return F.relu(input, inplace=inplace)


@backend_registry.register(
    OperatorType.GELU, BackendType.NATIVE, [DataType.FLOAT32, DataType.BFLOAT16]
)
class NativeGeLU(BackendBase):

    def execute(
        self,
        input: torch.Tensor,
        approximate: bool = False,
    ) -> torch.Tensor:
        return F.gelu(input, approximate=approximate)


@backend_registry.register(
    OperatorType.SILU, BackendType.NATIVE, [DataType.FLOAT32, DataType.BFLOAT16]
)
class NativeSiLU(BackendBase):

    def execute(
        self,
        input: torch.Tensor,
        inplace: bool = False,
    ) -> torch.Tensor:
        return F.silu(input, inplace=inplace)


@backend_registry.register(
    OperatorType.TANH, BackendType.NATIVE, [DataType.FLOAT32, DataType.BFLOAT16]
)
class NativeTanh(BackendBase):

    def execute(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tanh(input)
