# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************
# In /ZenDNN_PACE/tests run using python -m unittest ops.test_mlp_mlp_fusion.test_imbps

import torch
from torch.testing._internal.common_utils import TestCase
from hypothesis import given, settings, strategies as st
import pace  # noqa: F401


def block_tensor(tensor: torch.Tensor, block_dim: int, num_blocks: int) -> list:
    """
    Splits a 2-dimensional tensor into a list of contiguous sub-tensors (blocks) along the specified dimension.

    Args:
        tensor (torch.Tensor): The input 2-dimensional tensor to be blocked.
        block_dim (int): The dimension along which to block the tensor (0 or 1).
        num_blocks (int): The number of blocks to split the tensor into.

    Returns:
        list: A list of contiguous sub-tensors (blocks).
    """
    block_size = tensor.size(block_dim) // num_blocks
    blocks = torch.split(tensor, block_size, dim=block_dim)
    contiguous_blocks = [block.contiguous() for block in blocks]
    return contiguous_blocks


class dataSet:
    def __init__(self):
        self.M = 64
        self.K = 64
        self.N = 4 * self.K
        num_blocks = 4
        self.src = torch.rand(self.M, self.K, dtype=torch.float32) * 0.01
        self.weights1 = torch.rand(self.N, self.K, dtype=torch.float32) * 0.01
        self.bias1 = torch.rand(self.N, dtype=torch.float32) * 0.01
        self.weights2 = torch.rand(self.K, self.N, dtype=torch.float32) * 0.01
        self.bias2 = torch.rand(self.K, dtype=torch.float32) * 0.01
        self.weights_gateProj = torch.rand(self.N, self.K, dtype=torch.float32) * 0.01
        self.bias_gateProj = torch.rand(self.N, dtype=torch.float32) * 0.01
        self.weights1_blocks = block_tensor(self.weights1, 0, num_blocks)
        self.bias1_blocks = block_tensor(self.bias1, 0, num_blocks)
        self.weights2_blocks = block_tensor(self.weights2, 1, num_blocks)
        self.weights_gateProj_blocks = block_tensor(
            self.weights_gateProj, 0, num_blocks
        )
        self.bias_gateProj_blocks = block_tensor(self.bias_gateProj, 0, num_blocks)
        check = self.weights1.shape[0] // num_blocks == 0

        if check:
            self.weights1_blocks = torch.cat(
                (self.weights1_blocks[-1], self.weights1_blocks[-2]), dim=0
            )
            self.weights1_blocks = self.weights1_blocks[:-1]
            self.bias1_blocks = torch.cat(
                (self.bias1_blocks[-1], self.bias1_blocks[-2]), dim=0
            )
            self.bias1_blocks = self.bias1_blocks[:-1]
            self.weights2_blocks = torch.cat(
                (self.weights2_blocks[-1], self.weights2_blocks[-2]), dim=1
            )
            self.weights2_blocks = self.weights2_blocks[:-1]
            self.weights_gateProj_blocks = torch.cat(
                (self.weights_gateProj_blocks[-1], self.weights_gateProj_blocks[-2]),
                dim=0,
            )
            self.weights_gateProj_blocks = self.weights_gateProj_blocks[:-1]
            self.bias_gateProj_blocks = torch.cat(
                (self.bias_gateProj_blocks[-1], self.bias_gateProj_blocks[-2]),
                dim=0,
            )
        self.bias_gateProj_blocks = self.bias_gateProj_blocks[:-1]


_dataSet = dataSet()


def assertEqualWithTolerance(tensor1, tensor2, tolerance=0.01):
    diff = tensor1 - tensor2
    within_tolerance = torch.sum(torch.abs(diff) < 0.1).item()
    total_elements = torch.numel(tensor1)
    mismatch_percentage = (total_elements - within_tolerance) / total_elements * 100
    if mismatch_percentage < tolerance * 100:
        return True
    else:
        return False


class test_imbps(TestCase):
    def setUp(self, reuse_data=False):
        if reuse_data:
            self.src = _dataSet.src
            self.weights1 = _dataSet.weights1
            self.bias1 = _dataSet.bias1
            self.weights2 = _dataSet.weights2
            self.bias2 = _dataSet.bias2
            self.weights_gateProj = _dataSet.weights_gateProj
            self.bias_gateProj = _dataSet.bias_gateProj
            self.weights1_blocks = _dataSet.weights1_blocks
            self.bias1_blocks = _dataSet.bias1_blocks
            self.weights2_blocks = _dataSet.weights2_blocks
            self.weights_gateProj_blocks = _dataSet.weights_gateProj_blocks
            self.bias_gateProj_blocks = _dataSet.bias_gateProj_blocks

    @given(
        M=st.integers(min_value=1024, max_value=9216),
        K=st.integers(min_value=1024, max_value=8192),
        nlf=st.sampled_from(["Relu", "Gelu"]),
        dtype=st.sampled_from([torch.float32, torch.bfloat16]),
        num_blocks=st.integers(min_value=1, max_value=64),
    )
    @settings(deadline=None, max_examples=10)
    def test_imbps_opt(self, M, K, nlf, dtype, num_blocks):
        """
        Tests the MLP-MultiLayer Perceptron (MLP) fusion operation with block tensors
        specifically for OPT-based models.

        Args:
            M (int): Number of rows in the input tensor.
            K (int): Number of columns in the input tensor and rows in the weight tensors.
            nlf (str): Non-linear function to apply ('Relu', 'Gelu', or 'SiLU').
            dtype (torch.dtype): Data type of the tensors.
            num_blocks (int): Number of blocks to divide the weight and bias tensors into.
        """

        N = 4 * K
        self.src = torch.rand(M, K, dtype=dtype) * 0.01
        self.weights1 = torch.rand(N, K, dtype=dtype) * 0.01
        self.bias1 = torch.rand(N, dtype=dtype) * 0.01
        self.weights2 = torch.rand(K, N, dtype=dtype) * 0.01
        self.bias2 = torch.rand(K, dtype=dtype) * 0.01
        self.weights1_blocks = block_tensor(self.weights1, 0, num_blocks)
        self.bias1_blocks = block_tensor(self.bias1, 0, num_blocks)
        self.weights2_blocks = block_tensor(self.weights2, 1, num_blocks)
        check = self.weights1.shape[0] // num_blocks == 0

        if check:
            self.weights1_blocks = torch.cat(
                (self.weights1_blocks[-1], self.weights1_blocks[-2]), dim=0
            )
            self.weights1_blocks = self.weights1_blocks[:-1]
            self.bias1_blocks = torch.cat(
                (self.bias1_blocks[-1], self.bias1_blocks[-2]), dim=0
            )
            self.bias1_blocks = self.bias1_blocks[:-1]
            self.weights2_blocks = torch.cat(
                (self.weights2_blocks[-1], self.weights2_blocks[-2]), dim=0
            )
            self.weights2_blocks = self.weights2_blocks[:-1]
        NLF = torch.nn.ReLU() if nlf == "Relu" else torch.nn.GELU()
        result_fusion = torch.ops.pace.mlp_mlp_fusion(
            self.src,
            self.weights1_blocks,
            self.bias1_blocks,
            self.weights2_blocks,
            self.bias2,
            nlf,
            None,
            None,
        )
        result_reference = torch.nn.functional.linear(
            NLF(torch.nn.functional.linear(self.src, self.weights1, self.bias1)),
            self.weights2,
            self.bias2,
        )
        self.assertTrue(assertEqualWithTolerance(result_fusion, result_reference, 0.01))

    @given(
        M=st.integers(min_value=1024, max_value=9216),
        K=st.integers(min_value=1024, max_value=8192),
        dtype=st.sampled_from([torch.float32, torch.bfloat16]),
        num_blocks=st.integers(min_value=1, max_value=64),
    )
    @settings(deadline=None, max_examples=10)  # Reduced max_examples to 10
    def test_imbps_llama(self, M, K, dtype, num_blocks):
        """
        Tests the MLP-MultiLayer Perceptron (MLP) fusion operation with block tensors
        specifically for Llama-based models.
        Args:
            M (int): Number of rows in the input tensor.
            K (int): Number of columns in the input tensor and rows in the weight tensors.
            dtype (torch.dtype): Data type of the tensors.
            num_blocks (int): Number of blocks to divide the weight and bias tensors into.
        """
        nlf = "Silu"
        N = 4 * K
        self.src = torch.rand(M, K, dtype=dtype) * 0.01
        self.weights1 = torch.rand(N, K, dtype=dtype) * 0.01
        self.bias1 = torch.rand(N, dtype=dtype) * 0.01
        self.weights2 = torch.rand(K, N, dtype=dtype) * 0.01
        self.bias2 = torch.rand(K, dtype=dtype) * 0.01
        self.weights_gateProj = torch.rand(N, K, dtype=dtype) * 0.01
        self.bias_gateProj = torch.rand(N, dtype=dtype) * 0.01
        self.weights1_blocks = block_tensor(self.weights1, 0, num_blocks)
        self.bias1_blocks = block_tensor(self.bias1, 0, num_blocks)
        self.weights2_blocks = block_tensor(self.weights2, 1, num_blocks)
        self.weights_gateProj_blocks = block_tensor(
            self.weights_gateProj, 0, num_blocks
        )
        self.bias_gateProj_blocks = block_tensor(self.bias_gateProj, 0, num_blocks)
        check = self.weights1.shape[0] // num_blocks == 0
        if check:
            self.weights1_blocks = torch.cat(
                (self.weights1_blocks[-1], self.weights1_blocks[-2]), dim=0
            )
            self.weights1_blocks = self.weights1_blocks[:-1]
            self.bias1_blocks = torch.cat(
                (self.bias1_blocks[-1], self.bias1_blocks[-2]), dim=0
            )
            self.bias1_blocks = self.bias1_blocks[:-1]
            self.weights2_blocks = torch.cat(
                (self.weights2_blocks[-1], self.weights2_blocks[-2]), dim=0
            )
            self.weights2_blocks = self.weights2_blocks[:-1]
            self.weights_gateProj_blocks = torch.cat(
                (self.weights_gateProj_blocks[-1], self.weights_gateProj_blocks[-2]),
                dim=0,
            )
            self.weights_gateProj_blocks = self.weights_gateProj_blocks[:-1]
            self.bias_gateProj_blocks = torch.cat(
                (self.bias_gateProj_blocks[-1], self.bias_gateProj_blocks[-2]),
                dim=0,
            )
            self.bias_gateProj_blocks = self.bias_gateProj_blocks[:-1]
        NLF = torch.nn.SiLU()
        result_fusion = torch.ops.pace.mlp_mlp_fusion(
            self.src,
            self.weights1_blocks,
            self.bias1_blocks,
            self.weights2_blocks,
            self.bias2,
            nlf,
            self.weights_gateProj_blocks,
            self.bias_gateProj_blocks,
        )
        result_reference = torch.nn.functional.linear(
            (
                torch.nn.functional.linear(self.src, self.weights1, self.bias1)
                * NLF(
                    torch.nn.functional.linear(
                        self.src, self.weights_gateProj, self.bias_gateProj
                    )
                )
            ),
            self.weights2,
            self.bias2,
        )
        self.assertTrue(assertEqualWithTolerance(result_fusion, result_reference, 0.01))

    @given(
        M=st.integers(min_value=1024, max_value=9216),
        K=st.integers(min_value=1024, max_value=8192),
        nlf=st.sampled_from(["Relu", "Gelu"]),
        dtype=st.sampled_from([torch.float32, torch.bfloat16]),
        num_blocks=st.integers(min_value=1, max_value=64),
        extra_dims=st.lists(
            st.integers(min_value=1, max_value=10), min_size=1, max_size=3
        ),
    )
    @settings(deadline=None, max_examples=10)
    def test_imbps_opt_extra_dims(self, M, K, nlf, dtype, num_blocks, extra_dims):
        """
        Tests the MLP-MultiLayer Perceptron (MLP) fusion operation with block tensors
        specifically for OPT-based models with src tensor having more than 2 dimensions.
        Args:
            M (int): Number of rows in the input tensor.
            K (int): Number of columns in the input tensor and rows in the weight tensors.
            extra_dims (list): List of extra dimensions to add to the input tensor.
            nlf (str): Non-linear function to apply ('Relu', 'Gelu', or 'SiLU').
            dtype (torch.dtype): Data type of the tensors.
            num_blocks (int): Number of blocks to divide the weight and bias tensors into.
            extra_dims (list): List of extra dimensions to add to the input tensor.
        """
        N = 4 * K
        src_shape = extra_dims + [M, K]  # Adding random extra dimensions to src
        self.src = torch.rand(*src_shape, dtype=dtype)  # Making src multi-dimensional
        self.weights1 = torch.rand(N, K, dtype=dtype) * 0.01
        self.bias1 = torch.rand(N, dtype=dtype) * 0.01
        self.weights2 = torch.rand(K, N, dtype=dtype) * 0.01
        self.bias2 = torch.rand(K, dtype=dtype) * 0.01
        self.weights1_blocks = block_tensor(self.weights1, 0, num_blocks)
        self.bias1_blocks = block_tensor(self.bias1, 0, num_blocks)
        self.weights2_blocks = block_tensor(self.weights2, 1, num_blocks)
        check = self.weights1.shape[0] // num_blocks == 0

        if check:
            self.weights1_blocks = torch.cat(
                (self.weights1_blocks[-1], self.weights1_blocks[-2]), dim=0
            )
            self.weights1_blocks = self.weights1_blocks[:-1]
            self.bias1_blocks = torch.cat(
                (self.bias1_blocks[-1], self.bias1_blocks[-2]), dim=0
            )
            self.bias1_blocks = self.bias1_blocks[:-1]
            self.weights2_blocks = torch.cat(
                (self.weights2_blocks[-1], self.weights2_blocks[-2]), dim=0
            )
            self.weights2_blocks = self.weights2_blocks[:-1]
            self.weights_gateProj_blocks = torch.cat(
                (self.weights_gateProj_blocks[-1], self.weights_gateProj_blocks[-2]),
                dim=0,
            )
            self.weights_gateProj_blocks = self.weights_gateProj_blocks[:-1]
        result_fusion = torch.ops.pace.mlp_mlp_fusion(
            self.src,
            self.weights1_blocks,
            self.bias1_blocks,
            self.weights2_blocks,
            self.bias2,
            nlf,
            None,
            None,
        )
        NLF = torch.nn.GELU() if nlf == "Gelu" else torch.nn.ReLU()
        result_reference = torch.nn.functional.linear(
            NLF(torch.nn.functional.linear(self.src, self.weights1, self.bias1)),
            self.weights2,
            self.bias2,
        )
        self.assertTrue(assertEqualWithTolerance(result_fusion, result_reference, 0.01))

    @given(
        M=st.integers(min_value=1024, max_value=9216),
        K=st.integers(min_value=1024, max_value=8192),
        dtype=st.sampled_from([torch.float32, torch.bfloat16]),
        num_blocks=st.integers(min_value=1, max_value=64),
        extra_dims=st.lists(
            st.integers(min_value=1, max_value=10), min_size=1, max_size=3
        ),
    )
    @settings(deadline=None, max_examples=10)  # Reduced max_examples to 10
    def test_imbps_llama_extra_dims(self, M, K, dtype, num_blocks, extra_dims):
        """
        Tests the MLP-MultiLayer Perceptron (MLP) fusion operation with block tensors
        specifically for Llama-based models with src tensor having more than 2 dimensions.
        Args:
            M (int): Number of rows in the input tensor.
            K (int): Number of columns in the input tensor and rows in the weight tensors.
            dtype (torch.dtype): Data type of the tensors.
            num_blocks (int): Number of blocks to divide the weight and bias tensors into.
            extra_dims (list): List of extra dimensions to add to the input tensor.
        """
        nlf = "Silu"
        N = 4 * K
        self.src = torch.rand(M, K, dtype=dtype) * 0.01
        self.weights1 = torch.rand(N, K, dtype=dtype) * 0.01
        self.bias1 = torch.rand(N, dtype=dtype) * 0.01
        self.weights2 = torch.rand(K, N, dtype=dtype) * 0.01
        self.bias2 = torch.rand(K, dtype=dtype) * 0.01
        self.weights_gateProj = torch.rand(N, K, dtype=dtype) * 0.01
        self.bias_gateProj = torch.rand(N, dtype=dtype) * 0.01
        self.weights1_blocks = block_tensor(self.weights1, 0, num_blocks)
        self.bias1_blocks = block_tensor(self.bias1, 0, num_blocks)
        self.weights2_blocks = block_tensor(self.weights2, 1, num_blocks)
        self.weights_gateProj_blocks = block_tensor(
            self.weights_gateProj, 0, num_blocks
        )
        self.bias_gateProj_blocks = block_tensor(self.bias_gateProj, 0, num_blocks)
        check = self.weights1.shape[0] // num_blocks == 0
        if check:
            self.weights1_blocks = torch.cat(
                (self.weights1_blocks[-1], self.weights1_blocks[-2]), dim=0
            )
            self.weights1_blocks = self.weights1_blocks[:-1]
            self.bias1_blocks = torch.cat(
                (self.bias1_blocks[-1], self.bias1_blocks[-2]), dim=0
            )
            self.bias1_blocks = self.bias1_blocks[:-1]
            self.weights2_blocks = torch.cat(
                (self.weights2_blocks[-1], self.weights2_blocks[-2]), dim=0
            )
            self.weights2_blocks = self.weights2_blocks[:-1]
            self.weights_gateProj_blocks = torch.cat(
                (self.weights_gateProj_blocks[-1], self.weights_gateProj_blocks[-2]),
                dim=0,
            )
            self.weights_gateProj_blocks = self.weights_gateProj_blocks[:-1]
            self.bias_gateProj_blocks = torch.cat(
                (self.bias_gateProj_blocks[-1], self.bias_gateProj_blocks[-2]),
                dim=0,
            )
            self.bias_gateProj_blocks = self.bias_gateProj_blocks[:-1]
        NLF = torch.nn.SiLU()
        result_fusion = torch.ops.pace.mlp_mlp_fusion(
            self.src,
            self.weights1_blocks,
            self.bias1_blocks,
            self.weights2_blocks,
            self.bias2,
            nlf,
            self.weights_gateProj_blocks,
            self.bias_gateProj_blocks,
        )
        result_reference = torch.nn.functional.linear(
            (
                torch.nn.functional.linear(self.src, self.weights1, self.bias1)
                * NLF(
                    torch.nn.functional.linear(
                        self.src, self.weights_gateProj, self.bias_gateProj
                    )
                )
            ),
            self.weights2,
            self.bias2,
        )
        self.assertTrue(assertEqualWithTolerance(result_fusion, result_reference, 0.01))

    @settings(deadline=None, max_examples=10)
    def test_imbps_invalid_opt(self):
        """
        Tests the MLP-MultiLayer Perceptron (MLP) fusion operation with block tensors
        specifically for invalid data types in OPT-based models.
        """
        nlf = "Gelu"
        self.setUp(True)
        with self.assertRaisesRegex(
            RuntimeError,
            "pace::mlp_mlp_fusion only support the dtypes Bfloat16 and Float types for output",
        ):
            torch.ops.pace.mlp_mlp_fusion(
                self.src.to(torch.float64),
                self.weights1_blocks,
                self.bias1_blocks,
                self.weights2_blocks,
                self.bias2,
                nlf,
                None,
                None,
            )
        with self.assertRaisesRegex(
            RuntimeError,
            "pace::mlp_mlp_fusion only support the dtypes Bfloat16 and Float types for output",
        ):
            torch.ops.pace.mlp_mlp_fusion(
                self.src.to(torch.int8),
                self.weights1_blocks,
                self.bias1_blocks,
                self.weights2_blocks,
                self.bias2,
                nlf,
                None,
                None,
            )

    @settings(deadline=None, max_examples=10)
    def test_imbps_invalid_llama(self):
        """
        Tests the MLP-MultiLayer Perceptron (MLP) fusion operation with block tensors
        specifically for invalid data types in Llama-based models.
        """
        nlf = "Silu"
        self.setUp(True)

        with self.assertRaisesRegex(
            RuntimeError,
            "pace::mlp_mlp_fusion only support the dtypes Bfloat16 and Float types for output",
        ):
            torch.ops.pace.mlp_mlp_fusion(
                self.src.to(torch.float64),
                self.weights1_blocks,
                self.bias1_blocks,
                self.weights2_blocks,
                self.bias2,
                nlf,
                self.weights_gateProj_blocks,
                self.bias_gateProj_blocks,
            )
        with self.assertRaisesRegex(
            RuntimeError,
            "pace::mlp_mlp_fusion only support the dtypes Bfloat16 and Float types for output",
        ):
            torch.ops.pace.mlp_mlp_fusion(
                self.src.to(torch.int8),
                self.weights1_blocks,
                self.bias1_blocks,
                self.weights2_blocks,
                self.bias2,
                nlf,
                self.weights_gateProj_blocks,
                self.bias_gateProj_blocks,
            )

    @given(
        InvalidNLF=st.sampled_from(["sigmoid", "tanh"]),
    )
    @settings(deadline=None, max_examples=10)
    def test_imbps_opt_invalid_nlf(self, InvalidNLF):
        """
        Tests the MLP-MultiLayer Perceptron (MLP) fusion operation with block tensors
        specifically for OPT-based models with invalid non-linear function.
        """
        self.setUp(True)
        with self.assertRaisesRegex(
            RuntimeError,
            "pace::mlp_mlp_fusion only supports gelu and relu for opt type models, got "
            + InvalidNLF,
        ):
            torch.ops.pace.mlp_mlp_fusion(
                self.src,
                self.weights1_blocks,
                self.bias1_blocks,
                self.weights2_blocks,
                self.bias2,
                InvalidNLF,
                None,
                None,
            )

    @given(
        InvalidNLF=st.sampled_from(["sigmoid", "tanh", "relu", "gelu"]),
    )
    @settings(deadline=None, max_examples=10)
    def test_imbps_llama_invalid_nlf(self, InvalidNLF):
        """
        Tests the MLP-MultiLayer Perceptron (MLP) fusion operation with block tensors
        specifically for Llama-based models with invalid non-linear function.
        """
        self.setUp(True)
        with self.assertRaisesRegex(
            RuntimeError,
            "pace::mlp_mlp_fusion only supports silu for llama type models, got "
            + InvalidNLF,
        ):
            torch.ops.pace.mlp_mlp_fusion(
                self.src,
                self.weights1_blocks,
                self.bias1_blocks,
                self.weights2_blocks,
                self.bias2,
                InvalidNLF,
                self.weights_gateProj_blocks,
                None,
            )
