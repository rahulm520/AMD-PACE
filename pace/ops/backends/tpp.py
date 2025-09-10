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
from pace.ops.enum import OperatorType, FusedOperatorType, BackendType, DataType


@backend_registry.register(OperatorType.LINEAR, BackendType.TPP, [DataType.BFLOAT16])
class TPPLinear(BackendBase):

    def preprocess(self, layer):
        weight = layer.weight
        block_size = int(os.getenv("LIBXSMM_BLOCK_SIZE", 32))
        # For the optimized path, the weight should be reshaped to 5D tensor
        # with shape (M/block_size, block_size, N/64, 32, 2)
        if weight.size(0) % block_size == 0 and weight.size(1) % 64 == 0:
            weight = torch.reshape(
                weight,
                (weight.size(0) // block_size, block_size, weight.size(1) // 64, 32, 2),
            )
            layer.weight = nn.Parameter(
                torch.permute(weight, (0, 2, 3, 1, 4)).contiguous()
            )

    def preprocess_input(self, input: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the input tensor to ensure it has the correct shape.
        """
        self.orig_shape = input.shape[:-1]
        if input.dim() < 3:
            for _ in range(3 - input.dim()):
                input = input.unsqueeze(0)
        elif input.dim() > 3:
            input = input.reshape(-1, input.size(-1))
        return input

    def postprocess_output(self, output: torch.Tensor) -> torch.Tensor:
        """
        Postprocess the output tensor to restore the original shape.
        """
        output = output.reshape(*self.orig_shape, -1)
        return output

    def execute(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input = self.preprocess_input(input)
        output = torch.ops.pace.libxsmmlinear_plain(input, weight, bias)
        return self.postprocess_output(output)


@backend_registry.register(
    FusedOperatorType.FUSEDLINEARRELU, BackendType.TPP, [DataType.BFLOAT16]
)
class TPPFusedLinearRelu(TPPLinear):

    def preprocess(self, layer):
        super().preprocess(layer.linear)

    def execute(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input = self.preprocess_input(input)
        output = torch.ops.pace.libxsmmlinear_relu(input, weight, bias)
        return self.postprocess_output(output)


@backend_registry.register(
    FusedOperatorType.FUSEDLINEARGELU, BackendType.TPP, [DataType.BFLOAT16]
)
class TPPFusedLinearGelu(TPPLinear):

    def preprocess(self, layer):
        super().preprocess(layer.linear)

    def execute(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input = self.preprocess_input(input)
        output = torch.ops.pace.libxsmmlinear_gelu(input, weight, bias)
        return self.postprocess_output(output)


@backend_registry.register(
    FusedOperatorType.FUSEDLINEARSILU, BackendType.TPP, [DataType.BFLOAT16]
)
class TPPFusedLinearSilU(TPPLinear):

    def preprocess(self, layer):
        super().preprocess(layer.linear)

    def execute(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input = self.preprocess_input(input)
        output = torch.ops.pace.libxsmmlinear_silu(input, weight, bias)
        return self.postprocess_output(output)


@backend_registry.register(
    FusedOperatorType.FUSEDLINEARMUL, BackendType.TPP, [DataType.BFLOAT16]
)
class TPPFusedLinearMul(TPPLinear):

    def preprocess(self, layer):
        super().preprocess(layer.linear)

    def execute(
        self,
        input: torch.Tensor,
        mul: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input = self.preprocess_input(input)
        mul = self.preprocess_input(mul)
        output = torch.ops.pace.libxsmmlinear_mul(input, mul, weight, bias)
        return self.postprocess_output(output)
