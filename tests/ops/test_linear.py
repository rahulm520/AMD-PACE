# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************
from typing import Callable, Optional, Tuple

import pace  # noqa: F401
import torch
import torch.nn.functional as F
from hypothesis import given
from hypothesis import strategies as st
from torch.testing._internal.common_utils import TestCase

from test_utils import (
    quantize_per_channel,
    quantize_per_tensor,
    quantize_per_tensor_without_zeropoint,
)


def prepare_linear_input(
    ndim: int, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare inputs for linear tests

    Args:
        ndim: Number of dimensions for input tensor
        dtype: Data type of input tensor

    Returns:
        inputs: Input tensor
        weights: Weight tensor
        bias: Bias tensor
    """

    min_power = 1
    max_power = 10

    input_size = (
        2 ** torch.randint(low=min_power, high=max_power, size=(ndim,))
    ).tolist()
    weight_size = (
        2 ** torch.randint(low=min_power, high=max_power, size=(1,))
    ).tolist()
    weight_size.append(input_size[-1])

    inputs = torch.rand(input_size, dtype=dtype)
    weights = torch.rand(weight_size, dtype=dtype)
    bias = torch.rand(weight_size[0], dtype=dtype)

    return inputs, weights, bias


def prepare_qlinear_input(
    ndim: int,
    input_dtype: Optional[torch.dtype] = torch.quint8,
    weight_dtype: Optional[torch.dtype] = torch.qint8,
    weight_quantize_method: Optional[Callable] = quantize_per_channel,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, int]:
    """
    Prepare inputs for quantized linear tests.
    The reference is calculated to find the output scale and zero point

    Args:
        ndim: Number of dimensions for input tensor
        input_dtype: Data type of input tensor
        weight_dtype: Data type of weight tensor

    Returns:
        qinputs: Quantized input tensor
        qweights: Quantized weight tensor
        bias: Bias tensor
        o_scale: Output scale
        o_zero_point: Output zero point
    """

    min_power = 1
    max_power = 10

    input_size = (
        2 ** torch.randint(low=min_power, high=max_power, size=(ndim,))
    ).tolist()
    weight_size = (
        2 ** torch.randint(low=min_power, high=max_power, size=(1,))
    ).tolist()
    weight_size.append(input_size[-1])

    qinputs = quantize_per_tensor(torch.rand(input_size) - 0.5, input_dtype)
    qweights = weight_quantize_method(torch.rand(weight_size) - 0.5, weight_dtype)
    bias = torch.rand(weight_size[0]) - 0.5

    qoutputs = quantize_per_tensor(
        torch.nn.functional.linear(qinputs.dequantize(), qweights.dequantize(), bias),
        input_dtype,
    )
    o_scale, o_zero_point = qoutputs.q_scale(), qoutputs.q_zero_point()

    return qinputs, qweights, bias, o_scale, o_zero_point


class TestLinear(TestCase):
    """
    Test cases for linear linear operators
    """

    @given(
        ndim=st.integers(1, 3), dtype=st.sampled_from([torch.float32, torch.bfloat16])
    )
    def test_linear_nd(self, ndim, dtype):
        inputs, weights, bias = prepare_linear_input(ndim, dtype)
        std_out = torch.nn.functional.linear(inputs, weights, bias)
        pace_out = torch.ops.pace.linear(inputs, weights, bias)
        self.assertEqual(std_out, pace_out, atol=1e-3, rtol=1e-3)

    @given(
        ndim=st.integers(1, 3), dtype=st.sampled_from([torch.float32, torch.bfloat16])
    )
    def test_linear_nd_without_bias(self, ndim, dtype):
        inputs, weights, _ = prepare_linear_input(ndim, dtype)
        std_out = torch.nn.functional.linear(inputs, weights)
        pace_out = torch.ops.pace.linear(inputs, weights, None)
        self.assertEqual(std_out, pace_out, atol=1e-3, rtol=1e-3)

    @given(
        ndim=st.integers(1, 3), dtype=st.sampled_from([torch.float32, torch.bfloat16])
    )
    def test_linear_with_relu(self, ndim, dtype):
        inputs, weights, bias = prepare_linear_input(ndim, dtype)
        std_out = torch.relu(torch.nn.functional.linear(inputs, weights, bias))
        pace_out = torch.ops.pace.linear_relu(inputs, weights, bias)
        self.assertEqual(std_out, pace_out, atol=1e-3, rtol=1e-3)

    @given(
        ndim=st.integers(1, 3), dtype=st.sampled_from([torch.float32, torch.bfloat16])
    )
    def test_linear_with_relu_without_bias(self, ndim, dtype):
        inputs, weights, _ = prepare_linear_input(ndim, dtype)
        std_out = torch.relu(torch.nn.functional.linear(inputs, weights))
        pace_out = torch.ops.pace.linear_relu(inputs, weights, None)
        self.assertEqual(std_out, pace_out, atol=1e-3, rtol=1e-3)

    def test_linear_invalid_input(self):
        inputs, weights, bias = prepare_linear_input(1, torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "mismatched types"):
            torch.ops.pace.linear(inputs, weights.to(torch.float32), bias)

        with self.assertRaisesRegex(RuntimeError, "mismatched types"):
            torch.ops.pace.linear(inputs.to(torch.float32), weights, bias)

        with self.assertRaisesRegex(RuntimeError, "mismatched types"):
            torch.ops.pace.linear(inputs, weights, bias.to(torch.float32))

    def test_linear_invalid_input_shape(self):
        inputs, weights, bias = prepare_linear_input(1, torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "got incompatible weight and bias"):
            torch.ops.pace.linear(inputs, weights, bias[1:])

        with self.assertRaisesRegex(RuntimeError, "got incompatible input and weight"):
            torch.ops.pace.linear(inputs[1:], weights, bias)

        with self.assertRaisesRegex(RuntimeError, "got incompatible input and weight"):
            torch.ops.pace.linear(inputs[1:], weights[1:], bias[1:])

    def test_linear_invalid_input_dim(self):
        inputs, weights, bias = prepare_linear_input(1, torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "got dims less than 1"):
            torch.ops.pace.linear(inputs[0], weights, bias)

        with self.assertRaisesRegex(
            RuntimeError, "only supports 2D tensors for weight"
        ):
            torch.ops.pace.linear(inputs, weights[0], bias)

        with self.assertRaisesRegex(RuntimeError, "only supports 1D tensors for bias"):
            torch.ops.pace.linear(inputs, weights, bias.unsqueeze(0))

    def test_linear_invalid_input_dtype(self):
        inputs, weights, bias = prepare_linear_input(1, torch.bfloat16)
        with self.assertRaisesRegex(
            RuntimeError, "only support the dtypes float and bfloat16"
        ):
            torch.ops.pace.linear(
                inputs.to(torch.int32), weights.to(torch.int32), bias.to(torch.int32)
            )

    def test_linear_invalid_input_relu(self):
        inputs, weights, bias = prepare_linear_input(1, torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "mismatched types"):
            torch.ops.pace.linear_relu(inputs, weights.to(torch.float32), bias)

        with self.assertRaisesRegex(RuntimeError, "mismatched types"):
            torch.ops.pace.linear_relu(inputs.to(torch.float32), weights, bias)

        with self.assertRaisesRegex(RuntimeError, "mismatched types"):
            torch.ops.pace.linear_relu(inputs, weights, bias.to(torch.float32))

    def test_linear_invalid_input_shape_relu(self):
        inputs, weights, bias = prepare_linear_input(1, torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "got incompatible weight and bias"):
            torch.ops.pace.linear_relu(inputs, weights, bias[1:])

        with self.assertRaisesRegex(RuntimeError, "got incompatible input and weight"):
            torch.ops.pace.linear_relu(inputs[1:], weights, bias)

        with self.assertRaisesRegex(RuntimeError, "got incompatible input and weight"):
            torch.ops.pace.linear_relu(inputs[1:], weights[1:], bias[1:])

    def test_linear_invalid_input_dim_relu(self):
        inputs, weights, bias = prepare_linear_input(1, torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "got dims less than 1"):
            torch.ops.pace.linear_relu(inputs[0], weights, bias)

        with self.assertRaisesRegex(
            RuntimeError, "only supports 2D tensors for weight"
        ):
            torch.ops.pace.linear_relu(inputs, weights[0], bias)

        with self.assertRaisesRegex(RuntimeError, "only supports 1D tensors for bias"):
            torch.ops.pace.linear_relu(inputs, weights, bias.unsqueeze(0))

    def test_linear_invalid_input_dtype_relu(self):
        inputs, weights, bias = prepare_linear_input(1, torch.bfloat16)
        with self.assertRaisesRegex(
            RuntimeError, "only support the dtypes float and bfloat16"
        ):
            torch.ops.pace.linear_relu(
                inputs.to(torch.int32), weights.to(torch.int32), bias.to(torch.int32)
            )


WEIGHT_QUANTIZE_METHODS = [quantize_per_channel, quantize_per_tensor_without_zeropoint]


class TestQlinear(TestCase):
    """
    Test cases for quantized linear operators

    The quantized linear operators are tested against the reference implementation
    which is a simulated equivalent of the quantized linear operator.
    """

    @given(
        ndim=st.integers(2, 3), quantize_method=st.sampled_from(WEIGHT_QUANTIZE_METHODS)
    )
    def test_qlinear_nd(self, ndim, quantize_method):
        inputs, weights, bias, o_scale, o_zero_point = prepare_qlinear_input(
            ndim, weight_quantize_method=quantize_method
        )

        std_out = torch.quantize_per_tensor(
            F.linear(inputs.dequantize(), weights.dequantize(), bias),
            o_scale,
            o_zero_point,
            torch.quint8,
        ).dequantize()
        pace_out = torch.ops.pace.qlinear(
            inputs, weights, bias, o_scale, o_zero_point, torch.quint8
        ).dequantize()

        self.assertEqual(std_out, pace_out, atol=1e-3, rtol=1e-3)

    @given(
        ndim=st.integers(2, 3), quantize_method=st.sampled_from(WEIGHT_QUANTIZE_METHODS)
    )
    def test_qlinear_nd_without_bias(self, ndim, quantize_method):
        inputs, weights, bias, o_scale, o_zero_point = prepare_qlinear_input(
            ndim, weight_quantize_method=quantize_method
        )

        std_out = torch.quantize_per_tensor(
            F.linear(inputs.dequantize(), weights.dequantize()),
            o_scale,
            o_zero_point,
            torch.quint8,
        ).dequantize()
        pace_out = torch.ops.pace.qlinear(
            inputs, weights, None, o_scale, o_zero_point, torch.quint8
        ).dequantize()

        self.assertEqual(std_out, pace_out, atol=1e-3, rtol=1e-3)

    @given(
        ndim=st.integers(2, 3), quantize_method=st.sampled_from(WEIGHT_QUANTIZE_METHODS)
    )
    def test_qlinear_with_relu(self, ndim, quantize_method):
        inputs, weights, bias, o_scale, o_zero_point = prepare_qlinear_input(
            ndim, weight_quantize_method=quantize_method
        )

        std_out = torch.quantize_per_tensor(
            torch.relu(F.linear(inputs.dequantize(), weights.dequantize(), bias)),
            o_scale,
            o_zero_point,
            torch.quint8,
        ).dequantize()
        pace_out = torch.ops.pace.qlinear_relu(
            inputs, weights, bias, o_scale, o_zero_point, torch.quint8
        ).dequantize()

        self.assertEqual(std_out, pace_out, atol=1e-3, rtol=1e-3)

    @given(
        ndim=st.integers(2, 3), quantize_method=st.sampled_from(WEIGHT_QUANTIZE_METHODS)
    )
    def test_qlinear_with_relu_without_bias(self, ndim, quantize_method):
        inputs, weights, bias, o_scale, o_zero_point = prepare_qlinear_input(
            ndim, weight_quantize_method=quantize_method
        )

        std_out = torch.quantize_per_tensor(
            torch.relu(F.linear(inputs.dequantize(), weights.dequantize())),
            o_scale,
            o_zero_point,
            torch.quint8,
        ).dequantize()
        pace_out = torch.ops.pace.qlinear_relu(
            inputs, weights, None, o_scale, o_zero_point, torch.quint8
        ).dequantize()

        self.assertEqual(std_out, pace_out, atol=1e-3, rtol=1e-3)

    @given(
        ndim=st.integers(2, 3), quantize_method=st.sampled_from(WEIGHT_QUANTIZE_METHODS)
    )
    def test_qlinear_with_sigmoid(self, ndim, quantize_method):
        inputs, weights, bias, o_scale, o_zero_point = prepare_qlinear_input(
            ndim, weight_quantize_method=quantize_method
        )

        std_out = torch.sigmoid(
            F.linear(inputs.dequantize(), weights.dequantize(), bias)
        )
        pace_out = torch.ops.pace.qlinear_sigmoid(inputs, weights, bias)

        self.assertEqual(std_out, pace_out, atol=1e-3, rtol=1e-3)

    @given(
        ndim=st.integers(2, 3), quantize_method=st.sampled_from(WEIGHT_QUANTIZE_METHODS)
    )
    def test_qlinear_with_sigmoid_without_bias(self, ndim, quantize_method):
        inputs, weights, bias, o_scale, o_zero_point = prepare_qlinear_input(
            ndim, weight_quantize_method=quantize_method
        )

        std_out = torch.sigmoid(F.linear(inputs.dequantize(), weights.dequantize()))
        pace_out = torch.ops.pace.qlinear_sigmoid(inputs, weights, None)

        self.assertEqual(std_out, pace_out, atol=1e-3, rtol=1e-3)

    @given(quantize_method=st.sampled_from(WEIGHT_QUANTIZE_METHODS))
    def test_qlinear_with_mul_add(self, quantize_method):
        inputs, weights, bias, o_scale, o_zero_point = prepare_qlinear_input(
            2, weight_quantize_method=quantize_method
        )

        linear_out = F.linear(inputs.dequantize(), weights.dequantize(), bias)

        mul_input = torch.rand_like(linear_out)
        add_input = torch.rand_like(linear_out)
        std_out = (linear_out * mul_input) + add_input

        pace_out = torch.ops.pace.qlinear_mul_add(
            inputs, weights, bias, mul_input, add_input, 1
        )
        self.assertEqual(std_out, pace_out, atol=1e-3, rtol=1e-3)

    @given(quantize_method=st.sampled_from(WEIGHT_QUANTIZE_METHODS))
    def test_qlinear_with_mul_without_bias(self, quantize_method):
        inputs, weights, bias, o_scale, o_zero_point = prepare_qlinear_input(
            2, weight_quantize_method=quantize_method
        )

        linear_out = F.linear(inputs.dequantize(), weights.dequantize())

        mul_input = torch.rand_like(linear_out)
        add_input = torch.rand_like(linear_out)
        std_out = (linear_out * mul_input) + add_input

        pace_out = torch.ops.pace.qlinear_mul_add(
            inputs, weights, None, mul_input, add_input, 1
        )

        self.assertEqual(std_out, pace_out, atol=1e-3, rtol=1e-3)

    def test_qlinear_invalid_input_shape(self):
        inputs, weights, bias, o_scale, o_zero_point = prepare_qlinear_input(1)
        with self.assertRaisesRegex(RuntimeError, "got incompatible input and weight"):
            torch.ops.pace.qlinear(
                inputs[1:], weights, bias, o_scale, o_zero_point, torch.quint8
            )

        with self.assertRaisesRegex(RuntimeError, "got incompatible weight and bias"):
            torch.ops.pace.qlinear(
                inputs, weights[1:], bias, o_scale, o_zero_point, torch.quint8
            )

        with self.assertRaisesRegex(RuntimeError, "got incompatible input and weight"):
            torch.ops.pace.qlinear(
                inputs[1:], weights[1:], bias, o_scale, o_zero_point, torch.quint8
            )

    def test_qlinear_invalid_input_dtype(self):
        inputs, weights, bias, o_scale, o_zero_point = prepare_qlinear_input(2)
        with self.assertRaisesRegex(
            RuntimeError, "only support the dtypes Int8 and Float types for output"
        ):
            torch.ops.pace.qlinear(
                inputs, weights, bias, o_scale, o_zero_point, torch.float64
            )

        with self.assertRaisesRegex(
            RuntimeError, "only support the dtypes Int8 types for input"
        ):
            torch.ops.pace.qlinear(
                inputs.dequantize(), weights, bias, o_scale, o_zero_point, torch.quint8
            )

        with self.assertRaisesRegex(
            RuntimeError, "only support the dtypes QInt8 types for weights"
        ):
            torch.ops.pace.qlinear(
                inputs, weights.dequantize(), bias, o_scale, o_zero_point, torch.quint8
            )

        with self.assertRaisesRegex(
            RuntimeError, "only support the dtypes float types for bias"
        ):
            torch.ops.pace.qlinear(
                inputs,
                weights,
                bias.to(torch.int8),
                o_scale,
                o_zero_point,
                torch.quint8,
            )

        inputs, weights, bias, o_scale, o_zero_point = prepare_qlinear_input(
            2, weight_dtype=torch.quint8
        )
        with self.assertRaisesRegex(
            RuntimeError, "only support the dtypes QInt8 types for weights"
        ):
            torch.ops.pace.qlinear(
                inputs, weights, bias, o_scale, o_zero_point, torch.quint8
            )

    def test_qlinear_with_mul_add_invalid_input(self):
        inputs, weights, bias, _, _ = prepare_qlinear_input(2)
        mul_input = torch.rand(inputs.size(), dtype=torch.float32)
        add_input = torch.rand(inputs.size(), dtype=torch.float32)

        with self.assertRaisesRegex(
            RuntimeError, "only supports float type for multiplier"
        ):
            torch.ops.pace.qlinear_mul_add(
                inputs, weights, bias, mul_input.to(torch.int32), add_input, 1
            )

        with self.assertRaisesRegex(
            RuntimeError, "only supports float type for addend"
        ):
            torch.ops.pace.qlinear_mul_add(
                inputs, weights, bias, mul_input, add_input.to(torch.int32), 1
            )

        with self.assertRaisesRegex(
            RuntimeError, "only supports 2D tensors for multiplier"
        ):
            torch.ops.pace.qlinear_mul_add(
                inputs, weights, bias, mul_input.unsqueeze(0), add_input, 1
            )

        with self.assertRaisesRegex(
            RuntimeError, "only supports 2D tensors for addend"
        ):
            torch.ops.pace.qlinear_mul_add(
                inputs, weights, bias, mul_input, add_input.unsqueeze(0), 1
            )

        with self.assertRaisesRegex(RuntimeError, "only supports alpha=1"):
            torch.ops.pace.qlinear_mul_add(
                inputs, weights, bias, mul_input, add_input.unsqueeze(0), 2
            )
