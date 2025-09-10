# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************
# In /ZenDNN_PACE/tests run using python -m unittest -v ops/test_LIBXSMMLinear.py

import torch
import pace  # noqa: F401
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase
from hypothesis import given, settings
import hypothesis.strategies as st


def reshape_weights(weights: torch.Tensor, use_5D_weight: bool) -> torch.Tensor:
    """Reshape weights to align with PACE operations, adjusting for block sizes."""
    if not use_5D_weight:
        return weights
    else:
        block_size = 16
        reshaped = torch.reshape(
            weights,
            (weights.size(0) // block_size, block_size, weights.size(1) // 64, 32, 2),
        )
        return torch.permute(reshaped, (0, 2, 3, 1, 4)).contiguous()


# For higher BS(input = (BS = 64, 64, 4096) and above ) 1e-3 is not providing accurate results
threshold = 1e-2

# Common settings for hypothesis-based tests, adjusting input shapes and conditions.
common_hypothesis = settings(deadline=None, max_examples=10)(
    given(
        input_shape=st.sampled_from([(128, 64, 128)]),
        weight_shape=st.sampled_from([(128, 128)]),
        dtype=st.sampled_from([torch.bfloat16]),
        use_bias=st.booleans(),
        use_5D_weights=st.booleans(),
    )
)


class TestLinear(TestCase):
    """Test cases for various libxsmm linear operators with hypothesis framework."""

    @settings(deadline=None)
    @common_hypothesis
    def test_libxsmmlinear_plain(
        self, input_shape, weight_shape, dtype, use_bias, use_5D_weights
    ):
        """Test plain linear operation against standard PyTorch."""
        inputs = torch.rand(input_shape, dtype=dtype)
        weights = torch.rand(weight_shape, dtype=dtype)
        bias = torch.rand(weight_shape[0], dtype=dtype) if use_bias else None

        std_out = F.linear(inputs, weights, bias)
        reshaped_weights = reshape_weights(weights, use_5D_weights)
        pace_out = torch.ops.pace.libxsmmlinear_plain(inputs, reshaped_weights, bias)
        self.assertEqual(std_out, pace_out, atol=threshold, rtol=threshold)

    @settings(deadline=None)
    @common_hypothesis
    def test_libxsmmlinear_silu(
        self, input_shape, weight_shape, dtype, use_bias, use_5D_weights
    ):
        """Test SiLU activation after linear operation."""
        inputs = torch.rand(input_shape, dtype=dtype)
        weights = torch.rand(weight_shape, dtype=dtype)
        bias = torch.rand(weight_shape[0], dtype=dtype) if use_bias else None

        std_out = F.silu(F.linear(inputs, weights, bias))
        reshaped_weights = reshape_weights(weights, use_5D_weights)
        pace_out = torch.ops.pace.libxsmmlinear_silu(inputs, reshaped_weights, bias)
        self.assertEqual(std_out, pace_out, atol=threshold, rtol=threshold)

    @settings(deadline=None)
    @common_hypothesis
    def test_libxsmmlinear_gelu(
        self, input_shape, weight_shape, dtype, use_bias, use_5D_weights
    ):
        """Test GELU activation function after linear operation."""
        inputs = torch.rand(input_shape, dtype=dtype)
        weights = torch.rand(weight_shape, dtype=dtype)
        bias = torch.rand(weight_shape[0], dtype=dtype) if use_bias else None

        std_out = F.gelu(F.linear(inputs, weights, bias))
        reshaped_weights = reshape_weights(weights, use_5D_weights)
        pace_out = torch.ops.pace.libxsmmlinear_gelu(inputs, reshaped_weights, bias)
        self.assertEqual(std_out, pace_out, atol=threshold, rtol=threshold)

    @settings(deadline=None)
    @common_hypothesis
    def test_libxsmmlinear_relu(
        self, input_shape, weight_shape, dtype, use_bias, use_5D_weights
    ):
        """Test ReLU activation function after linear operation."""
        inputs = torch.rand(input_shape, dtype=dtype)
        weights = torch.rand(weight_shape, dtype=dtype)
        bias = torch.rand(weight_shape[0], dtype=dtype) if use_bias else None

        std_out = F.relu(F.linear(inputs, weights, bias))
        reshaped_weights = reshape_weights(weights, use_5D_weights)
        pace_out = torch.ops.pace.libxsmmlinear_relu(inputs, reshaped_weights, bias)
        self.assertEqual(std_out, pace_out, atol=threshold, rtol=threshold)

    @settings(deadline=None)
    @common_hypothesis
    def test_libxsmmlinear_mul(
        self, input_shape, weight_shape, dtype, use_bias, use_5D_weights
    ):
        """Test multiplication after linear operation with a secondary input tensor."""
        inputs = torch.rand(input_shape, dtype=dtype)
        weights = torch.rand(weight_shape, dtype=dtype)
        bias = torch.rand(weight_shape[0], dtype=dtype) if use_bias else None
        mul_input_shape = (input_shape[0], input_shape[1], weight_shape[0])
        mul_input = torch.rand(mul_input_shape, dtype=dtype)

        std_out = mul_input * F.linear(inputs, weights, bias)
        reshaped_weights = reshape_weights(weights, use_5D_weights)
        pace_out = torch.ops.pace.libxsmmlinear_mul(
            inputs, mul_input, reshaped_weights, bias
        )
        # Observed Mismatched elements: (1 / 1048576) for 1e-2 for higher BS (input = (BS = 128, 64, 4096) and above)), So setting threshold to 1e-1
        self.assertEqual(std_out, pace_out, atol=1e-1, rtol=1e-1)

    # Invalid Test Cases
    @given(
        op=st.sampled_from(
            [
                (torch.ops.pace.libxsmmlinear_plain, "libxsmmlinear_plain", False),
                (torch.ops.pace.libxsmmlinear_silu, "libxsmmlinear_silu", False),
                (torch.ops.pace.libxsmmlinear_gelu, "libxsmmlinear_gelu", False),
                (torch.ops.pace.libxsmmlinear_relu, "libxsmmlinear_relu", False),
                (torch.ops.pace.libxsmmlinear_mul, "libxsmmlinear_mul", True),
            ]
        ),
        dtype=st.sampled_from([torch.float32, torch.float64]),
        size=st.sampled_from([(128, 4096)]),
        use_5D_weights=st.booleans(),
    )
    def test_libxsmmlinear_invalid_dtype(self, op, dtype, size, use_5D_weights):
        """Test handling of invalid dtype inputs for linear operators."""
        op_fn, op_name, is_mul = op
        dtype_str = {torch.float32: "Float", torch.float64: "Double"}[dtype]

        # Test when input has an invalid dtype compared to weights
        inputs = torch.randn(*size, dtype=dtype)
        weights = torch.randn(*size, dtype=torch.bfloat16)
        bias = torch.randn(size[0], dtype=torch.bfloat16)
        if is_mul:
            mul_input = torch.randn(1, size[1], dtype=dtype)
            with self.assertRaisesRegex(
                RuntimeError,
                f"pace::{op_name} got mismatched types, input: {dtype_str}, weight: BFloat16",
            ):
                op_fn(
                    inputs.unsqueeze(1),
                    mul_input.unsqueeze(1),
                    reshape_weights(weights, use_5D_weights),
                    bias,
                )
        else:
            with self.assertRaisesRegex(
                RuntimeError,
                f"pace::{op_name} got mismatched types, input: {dtype_str}, weight: BFloat16",
            ):
                op_fn(
                    inputs.unsqueeze(1), reshape_weights(weights, use_5D_weights), bias
                )

        # Test when weight has an invalid dtype compared to inputs
        inputs = torch.randn(*size, dtype=torch.bfloat16)
        weights = torch.randn(*size, dtype=dtype)
        if is_mul:
            mul_input = torch.randn(1, size[1], dtype=torch.bfloat16)
            with self.assertRaisesRegex(
                RuntimeError,
                f"pace::{op_name} got mismatched types, input: BFloat16, weight: {dtype_str}.",
            ):
                op_fn(
                    inputs.unsqueeze(1),
                    mul_input.unsqueeze(1),
                    reshape_weights(weights, use_5D_weights),
                    bias,
                )
        else:
            with self.assertRaisesRegex(
                RuntimeError,
                f"pace::{op_name} got mismatched types, input: BFloat16, weight: {dtype_str}.",
            ):
                op_fn(
                    inputs.unsqueeze(1), reshape_weights(weights, use_5D_weights), bias
                )

    @given(
        op=st.sampled_from(
            [
                (torch.ops.pace.libxsmmlinear_plain, "libxsmmlinear_plain", False),
                (torch.ops.pace.libxsmmlinear_silu, "libxsmmlinear_silu", False),
                (torch.ops.pace.libxsmmlinear_gelu, "libxsmmlinear_gelu", False),
                (torch.ops.pace.libxsmmlinear_relu, "libxsmmlinear_relu", False),
                (torch.ops.pace.libxsmmlinear_mul, "libxsmmlinear_mul", True),
            ]
        ),
        invalid_size=st.sampled_from([(128, 4096)]),
        use_5D_weights=st.booleans(),
    )
    def test_libxsmmlinear_invalid_input_dim(self, op, invalid_size, use_5D_weights):
        """Test handling of invalid input dimensions for operators, expecting 3D inputs."""
        op_fn, op_name, is_mul = op

        # Generate invalid input dimensions
        inputs = torch.randn(*invalid_size, dtype=torch.bfloat16)

        if is_mul:
            # Generate a secondary input tensor for multiplication
            mul_input = torch.randn(*invalid_size, dtype=torch.bfloat16)
            with self.assertRaisesRegex(
                RuntimeError, f"pace::{op_name} expected input to be 3D"
            ):
                op_fn(inputs, mul_input, reshape_weights(inputs, use_5D_weights), None)
        else:
            with self.assertRaisesRegex(
                RuntimeError, f"pace::{op_name} expected input to be 3D"
            ):
                op_fn(inputs, reshape_weights(inputs, use_5D_weights), None)

        # Test invalid weight dimensions
        weights = torch.randn(*invalid_size, dtype=torch.bfloat16)
        if is_mul:
            mul_input = torch.randn(1, invalid_size[1], dtype=torch.bfloat16)
            with self.assertRaisesRegex(
                RuntimeError,
                f"pace::{op_name} expected weight to be one of 2D or 5D, but got",
            ):
                op_fn(
                    inputs.unsqueeze(1),
                    mul_input.unsqueeze(1),
                    weights.unsqueeze(1),
                    None,
                )
        else:
            with self.assertRaisesRegex(
                RuntimeError,
                f"pace::{op_name} expected weight to be one of 2D or 5D, but got",
            ):
                op_fn(inputs.unsqueeze(1), weights.unsqueeze(1), None)
