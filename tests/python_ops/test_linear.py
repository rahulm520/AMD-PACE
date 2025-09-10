# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from hypothesis import given
from hypothesis import strategies as st
from torch.testing._internal.common_utils import TestCase

import torch
import torch.nn.functional as F

from pace.ops.enum import OperatorType
from pace.ops.registry import backend_registry
from pace.ops.linear import Linear, RepeatedKVLinear


class TestLinear(TestCase):

    @given(
        st.sampled_from(backend_registry.get_available_backends(OperatorType.LINEAR))
    )
    def test_linear(self, backend):
        linear = Linear(128, 64, backend_impl=backend[0], dtype=backend[1])
        self.assertEqual(linear.weight.shape, (64, 128))
        self.assertEqual(linear.bias.shape, (64,))

        # Initialize weights and bias
        weight = torch.randn(64, 128, dtype=backend[1].to_torch_dtype())
        bias = torch.randn(64, dtype=backend[1].to_torch_dtype())
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)

        random_input = torch.randn(5, 128, dtype=backend[1].to_torch_dtype())
        ref_out = F.linear(random_input, weight, bias)

        linear.backend.preprocess(linear)
        output = linear(random_input)
        self.assertEqual(output.shape, (5, 64))
        self.assertEqual(output, ref_out)

    @given(
        st.sampled_from(backend_registry.get_available_backends(OperatorType.LINEAR))
    )
    def test_repeated_kv_linear(self, backend):
        num_key_value_heads = 8
        kv_linear = RepeatedKVLinear(
            64,
            128,
            num_key_value_heads=num_key_value_heads,
            backend_impl=backend[0],
            dtype=backend[1],
        )

        kv_linear.weight.load_weights(
            kv_linear.weight,
            torch.randn(
                128 // num_key_value_heads, 64, dtype=backend[1].to_torch_dtype()
            ),
        )
        self.assertEqual(kv_linear.weight.shape, (128, 64))

        kv_linear.bias.load_weights(
            kv_linear.bias,
            torch.randn(128 // num_key_value_heads, dtype=backend[1].to_torch_dtype()),
        )
        self.assertEqual(kv_linear.bias.shape, (128,))
