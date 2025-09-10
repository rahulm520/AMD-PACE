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

from pace.ops.enum import FusedOperatorType
from pace.ops.registry import backend_registry
from pace.ops.fused_linear import (
    FusedLinearGelu,
    FusedLinearMul,
    FusedLinearRelu,
    FusedLinearSiLU,
)

tol_theshold = 1e-2


class TestFusedLinear(TestCase):

    @given(
        st.sampled_from(
            backend_registry.get_available_backends(FusedOperatorType.FUSEDLINEARRELU)
        )
    )
    def test_fused_linear_relu(self, backend):

        fused_linear = FusedLinearRelu(
            128, 64, backend_impl=backend[0], dtype=backend[1]
        )
        self.assertEqual(fused_linear.linear.weight.shape, (64, 128))
        self.assertEqual(fused_linear.linear.bias.shape, (64,))
        self.assertIsNotNone(fused_linear.activation)

        weight = torch.randn(64, 128, dtype=backend[1].to_torch_dtype())
        bias = torch.randn(64, dtype=backend[1].to_torch_dtype())
        fused_linear.linear.weight.copy_(weight)
        fused_linear.linear.bias.copy_(bias)
        fused_linear.backend.preprocess(fused_linear)

        input_tensor = torch.randn(10, 128, dtype=backend[1].to_torch_dtype())
        output = fused_linear(input_tensor)

        expected_output = F.relu(F.linear(input_tensor, weight, bias))
        self.assertEqual(output, expected_output, rtol=tol_theshold, atol=tol_theshold)

    @given(
        st.sampled_from(
            backend_registry.get_available_backends(FusedOperatorType.FUSEDLINEARGELU)
        )
    )
    def test_fused_linear_gelu(self, backend):

        fused_linear = FusedLinearGelu(
            128, 64, backend_impl=backend[0], dtype=backend[1]
        )
        self.assertEqual(fused_linear.linear.weight.shape, (64, 128))
        self.assertEqual(fused_linear.linear.bias.shape, (64,))
        self.assertIsNotNone(fused_linear.activation)

        weight = torch.randn(64, 128, dtype=backend[1].to_torch_dtype())
        bias = torch.randn(64, dtype=backend[1].to_torch_dtype())
        fused_linear.linear.weight.copy_(weight)
        fused_linear.linear.bias.copy_(bias)
        fused_linear.backend.preprocess(fused_linear)

        input_tensor = torch.randn(10, 128, dtype=backend[1].to_torch_dtype())
        output = fused_linear(input_tensor)

        expected_output = F.gelu(F.linear(input_tensor, weight, bias))
        self.assertEqual(output, expected_output, rtol=tol_theshold, atol=tol_theshold)

    @given(
        st.sampled_from(
            backend_registry.get_available_backends(FusedOperatorType.FUSEDLINEARSILU)
        )
    )
    def test_fused_linear_silu(self, backend):

        fused_linear = FusedLinearSiLU(
            128, 64, backend_impl=backend[0], dtype=backend[1]
        )
        self.assertEqual(fused_linear.linear.weight.shape, (64, 128))
        self.assertEqual(fused_linear.linear.bias.shape, (64,))
        self.assertIsNotNone(fused_linear.activation)

        weight = torch.randn(64, 128, dtype=backend[1].to_torch_dtype())
        bias = torch.randn(64, dtype=backend[1].to_torch_dtype())
        fused_linear.linear.weight.copy_(weight)
        fused_linear.linear.bias.copy_(bias)
        fused_linear.backend.preprocess(fused_linear)

        input_tensor = torch.randn(10, 128, dtype=backend[1].to_torch_dtype())
        output = fused_linear(input_tensor)

        expected_output = F.silu(F.linear(input_tensor, weight, bias))
        self.assertEqual(output, expected_output, rtol=tol_theshold, atol=tol_theshold)

    @given(
        st.sampled_from(
            backend_registry.get_available_backends(FusedOperatorType.FUSEDLINEARMUL)
        )
    )
    def test_fused_linear_mul(self, backend):

        fused_linear = FusedLinearMul(
            128, 64, backend_impl=backend[0], dtype=backend[1]
        )
        self.assertEqual(fused_linear.linear.weight.shape, (64, 128))
        self.assertEqual(fused_linear.linear.bias.shape, (64,))

        weight = torch.randn(64, 128, dtype=backend[1].to_torch_dtype())
        bias = torch.randn(64, dtype=backend[1].to_torch_dtype())
        fused_linear.linear.weight.copy_(weight)
        fused_linear.linear.bias.copy_(bias)
        fused_linear.backend.preprocess(fused_linear)

        input_tensor = torch.randn(10, 128, dtype=backend[1].to_torch_dtype())
        mul_tensor = torch.randn(10, 64, dtype=backend[1].to_torch_dtype())
        output = fused_linear(input_tensor, mul_tensor)

        expected_output = F.linear(input_tensor, weight, bias) * mul_tensor
        # Fused operators can have slight numerical differences
        self.assertEqual(output, expected_output, rtol=1e-2, atol=1e-2)
