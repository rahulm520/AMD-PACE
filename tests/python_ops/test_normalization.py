# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from hypothesis import given
from hypothesis import strategies as st

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase


from pace.ops.enum import OperatorType
from pace.ops.registry import backend_registry
from pace.ops.normalization import LayerNorm, RMSNorm


class TestNormalization(TestCase):

    @given(
        st.sampled_from(backend_registry.get_available_backends(OperatorType.LAYERNORM))
    )
    def test_layernorm(self, backend):

        normalized_shape = (64,)
        layernorm = LayerNorm(
            normalized_shape,
            elementwise_affine=True,
            backend_impl=backend[0],
            dtype=backend[1],
        )
        self.assertEqual(layernorm.weight.shape, (64,))
        self.assertEqual(layernorm.bias.shape, (64,))

        weight = torch.randn(normalized_shape, dtype=backend[1].to_torch_dtype())
        bias = torch.randn(normalized_shape, dtype=backend[1].to_torch_dtype())
        layernorm.weight.copy_(weight)
        layernorm.bias.copy_(bias)

        x = torch.randn(1, 64, dtype=backend[1].to_torch_dtype())
        y = layernorm(x)
        y_torch = F.layer_norm(x, normalized_shape, weight=weight, bias=bias)
        self.assertEqual(y, y_torch)

    @given(
        st.sampled_from(backend_registry.get_available_backends(OperatorType.RMSNORM))
    )
    def test_rmsnorm(self, backend):

        normalized_shape = 64
        rmsnorm = RMSNorm(normalized_shape, backend_impl=backend[0], dtype=backend[1])
        self.assertEqual(rmsnorm.weight.shape, (64,))

        weight = torch.randn(normalized_shape, dtype=backend[1].to_torch_dtype())
        rmsnorm.weight.copy_(weight)

        x = torch.randn(1, 64, dtype=backend[1].to_torch_dtype())
        y = rmsnorm(x)
        y_torch = F.rms_norm(x, normalized_shape=(normalized_shape,), weight=weight)
        self.assertEqual(y, y_torch)
