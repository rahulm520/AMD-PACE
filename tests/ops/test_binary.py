# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import Tuple

import torch
from torch.testing._internal.common_utils import TestCase

import pace  # noqa: F401
from test_utils import quantize_per_tensor


def quantize_if_qdtype(
    tensor: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Quantize the tensor if the dtype is not float32 or bfloat16
    Assumption is that if it's not float32 or bfloat16, it's a quantized dtype

    Args:
        tensor: input tensor
        dtype: dtype of the tensor

    Returns:
        tensor: quantized tensor
    """

    if dtype in [torch.float32, torch.bfloat16]:
        return tensor.to(dtype)
    return quantize_per_tensor(tensor, dtype)


def prepare_inputs(
    qa_type: torch.dtype = torch.float32,
    qb_type: torch.dtype = torch.quint8,
    qaddend_type: torch.dtype = torch.quint8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, int]:
    """
    Prepare inputs for the test

    Args:
        qa_type: dtype of the first input
        qb_type: dtype of the second input
        qaddend_type: dtype of the addend

    Returns:
        qa: first input tensor
        qb: second input tensor
        qaddend: addend tensor
        ref_out: reference output tensor
        o_scale: output scale
        o_zero_point: output zero point
    """
    min_power = 1
    max_power = 50

    input_size = torch.randint(low=min_power, high=max_power, size=(2,)).tolist()
    input_size[0] = input_size[0] * torch.randint(low=1, high=100, size=(1,))
    input_size[1] = input_size[1] * 96  # second dimension is always a multiple of 96

    qa = torch.rand(input_size) - 0.5
    qa = quantize_if_qdtype(qa, qa_type)

    qb = torch.rand(input_size) - 0.5
    qb = quantize_if_qdtype(qb, qb_type)

    qaddend = torch.rand(input_size) - 0.5
    qaddend = quantize_if_qdtype(qaddend, qaddend_type)

    # using dequantize for all since without checks since if
    # it's not quantized, it will return the same tensor
    ref_out = qa.dequantize() * qb.dequantize() + qaddend.dequantize()
    qref = quantize_per_tensor(ref_out, torch.quint8)
    o_scale, o_zero_point = qref.q_scale(), qref.q_zero_point()

    return (qa, qb, qaddend, ref_out, o_scale, o_zero_point)


class TestQMulAdd(TestCase):
    """
    Test cases for pace::qmul_add
    The operator pace::qmul_add is a fused operator which is
    interal to the PACE library. It is a fusion of the quantized
    multiplication and addition operators. The pattern
    quantized::mul + quantized::add is automatically replaced
    with pace::qmul_add in the graph mode.
    """

    def test_mul_add(
        self,
    ):

        for addend_dtype in [torch.float32, torch.quint8]:
            qa, qb, qaddend, ref_out, o_scale, o_zero_point = prepare_inputs(
                qa_type=torch.float32, qb_type=torch.quint8, qaddend_type=addend_dtype
            )

            pace_out = torch.ops.pace.qmul_add(
                qa, qb, qaddend, o_scale, o_zero_point, torch.quint8
            ).dequantize()

            # Using a higher rate of error due to the way the addend is added
            self.assertEqual(ref_out, pace_out, atol=1e-2, rtol=1e-2)

    def test_invalid_dtypes(
        self,
    ):

        with self.assertRaisesRegex(
            RuntimeError, "only supports Float Tensor for A input"
        ):
            qa, qb, qaddend, _, o_scale, o_zero_point = prepare_inputs(
                qa_type=torch.quint8
            )
            torch.ops.pace.qmul_add(
                qa, qb, qaddend, o_scale, o_zero_point, torch.quint8
            )

        with self.assertRaisesRegex(
            RuntimeError, "only supports QUInt8 Tensor for B input"
        ):
            qa, qb, qaddend, _, o_scale, o_zero_point = prepare_inputs(
                qb_type=torch.float32
            )
            torch.ops.pace.qmul_add(
                qb, qa, qaddend, o_scale, o_zero_point, torch.quint8
            )

        with self.assertRaisesRegex(
            RuntimeError, "only support QUInt8 and Float Tensor for Addend input"
        ):
            qa, qb, qaddend, _, o_scale, o_zero_point = prepare_inputs(
                qaddend_type=torch.bfloat16
            )
            torch.ops.pace.qmul_add(
                qa, qb, qaddend, o_scale, o_zero_point, torch.quint8
            )

        with self.assertRaisesRegex(
            RuntimeError, "Expected a value of type 'number' for argument 'o_scale'"
        ):
            qa, qb, qaddend, _, o_scale, o_zero_point = prepare_inputs()
            torch.ops.pace.qmul_add(
                qb, qb, qaddend, torch.tensor([o_scale]), o_zero_point, torch.quint8
            )

        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a value of type 'number' for argument 'o_zero_point'",
        ):
            qa, qb, qaddend, _, o_scale, o_zero_point = prepare_inputs()
            torch.ops.pace.qmul_add(
                qb, qb, qaddend, o_scale, torch.tensor([o_zero_point]), torch.quint8
            )

    def test_invalid_shape(
        self,
    ):

        qa, qb, qaddend, _, o_scale, o_zero_point = prepare_inputs()

        with self.assertRaisesRegex(RuntimeError, expected_regex="same shape"):
            torch.ops.pace.qmul_add(
                qa, qb, qaddend[1:], o_scale, o_zero_point, torch.quint8
            )

        with self.assertRaisesRegex(RuntimeError, "same shape"):
            torch.ops.pace.qmul_add(
                qa, qb[1:], qaddend, o_scale, o_zero_point, torch.quint8
            )

        with self.assertRaisesRegex(RuntimeError, "same shape"):
            torch.ops.pace.qmul_add(
                qa[1:], qb, qaddend, o_scale, o_zero_point, torch.quint8
            )

        with self.assertRaisesRegex(RuntimeError, "same shape"):
            torch.ops.pace.qmul_add(
                qa, qb, qaddend.unsqueeze(0), o_scale, o_zero_point, torch.quint8
            )

        with self.assertRaisesRegex(
            RuntimeError, "requires 2nd dimension to be a factor of 96."
        ):
            torch.ops.pace.qmul_add(
                qa[:, :23],
                qb[:, :23],
                qaddend[:, :23],
                o_scale,
                o_zero_point,
                torch.quint8,
            )

    def test_invalid_dim(self):

        qa, qb, qaddend, _, o_scale, o_zero_point = prepare_inputs()
        with self.assertRaisesRegex(RuntimeError, "only supports 2D inputs"):
            torch.ops.pace.qmul_add(
                qa.unsqueeze(0),
                qb.unsqueeze(0),
                qaddend.unsqueeze(0),
                o_scale,
                o_zero_point,
                torch.quint8,
            )

        with self.assertRaisesRegex(RuntimeError, "only supports 2D inputs"):
            torch.ops.pace.qmul_add(
                qa[0], qb[0], qaddend[0], o_scale, o_zero_point, torch.quint8
            )
