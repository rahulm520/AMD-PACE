# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from hypothesis import given, settings
from hypothesis import strategies as st
from torch.testing._internal.common_utils import TestCase
import torch
from pace.ops.registry import backend_registry
from pace.ops.mlp import MergedMLP
from pace.ops.enum import FusedOperatorType


def assertEqualWithTolerance(tensor1, tensor2, tolerance=0.1):
    diff = tensor1 - tensor2
    within_tolerance = torch.sum(torch.abs(diff) < 0.1).item()
    total_elements = torch.numel(tensor1)
    mismatch_percentage = (total_elements - within_tolerance) / total_elements * 100
    if mismatch_percentage < tolerance * 100:
        return True
    else:
        return False


class TestMergedMLP(TestCase):
    @given(
        backend=st.sampled_from(
            backend_registry.get_available_backends(FusedOperatorType.FUSEDMLPLINEAR)
        ),
        bias=st.booleans(),
        activation=st.sampled_from(["relu", "gelu"]),
        gate=st.booleans(),
    )
    @settings(deadline=None, max_examples=10)
    def test_merged_mlp(self, backend, bias, activation, gate):
        if gate:
            activation = "silu"
        mlp = MergedMLP(
            in_features=1024,
            out_features=4096,
            bias=bias,
            activation=activation,
            gate=gate,
            dtype=backend[1],
            backend_impl=backend[0],
        )
        input_tensor = torch.randn(1024, 1024, dtype=backend[1].to_torch_dtype()) * 0.1
        bias_up = (
            torch.randn(4096, dtype=backend[1].to_torch_dtype()) * 0.1 if bias else None
        )
        weight_up = torch.randn(4096, 1024, dtype=backend[1].to_torch_dtype()) * 0.1
        bias_down = (
            torch.randn(1024, dtype=backend[1].to_torch_dtype()) * 0.1 if bias else None
        )
        weight_down = torch.randn(1024, 4096, dtype=backend[1].to_torch_dtype()) * 0.1
        if gate:
            weight_gate = (
                torch.randn(4096, 1024, dtype=backend[1].to_torch_dtype()) * 0.1
            )
            bias_gate = (
                torch.randn(4096, dtype=backend[1].to_torch_dtype()) * 0.1
                if bias
                else None
            )
        mlp.up_proj.linear.weight.copy_(weight_up)
        mlp.down_proj.weight.copy_(weight_down)
        mlp.gate_proj.linear.weight.copy_(weight_gate) if gate else None
        if bias:
            mlp.up_proj.linear.bias.copy_(bias_up)
            mlp.down_proj.bias.copy_(bias_down)
        if gate and bias:
            mlp.gate_proj.linear.bias.copy_(bias_gate)
        if activation == "silu":
            NLF = torch.nn.SiLU()
        elif activation == "relu":
            NLF = torch.nn.ReLU()
        elif activation == "gelu":
            NLF = torch.nn.GELU()

        if gate:
            result_reference = torch.nn.functional.linear(
                (
                    (torch.nn.functional.linear(input_tensor, weight_up, bias_up))
                    * NLF(
                        torch.nn.functional.linear(input_tensor, weight_gate, bias_gate)
                    )
                ),
                weight_down,
                bias_down,
            )
        else:
            result_reference = torch.nn.functional.linear(
                NLF(torch.nn.functional.linear(input_tensor, weight_up, bias_up)),
                weight_down,
                bias_down,
            )
        mlp.backend.preprocess(mlp)
        result_fusion = mlp(input_tensor)
        self.assertTrue(
            assertEqualWithTolerance(result_fusion, result_reference, tolerance=0.01)
        )
