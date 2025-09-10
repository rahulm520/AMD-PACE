# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from hypothesis import given
from hypothesis import strategies as st
from torch.testing._internal.common_utils import TestCase

import torch

from pace.ops.enum import OperatorType
from pace.ops.registry import backend_registry
from pace.ops.attention import MultiHeadAttention


class TestAttention(TestCase):

    @given(st.sampled_from(backend_registry.get_available_backends(OperatorType.MHA)))
    def test_sdpa(self, backend):
        mha = MultiHeadAttention(backend_impl=backend[0], dtype=backend[1])
        # Use torch's scaled_dot_product_attention for reference
        batch, seq, head, dim = 2, 4, 8, 16
        q = torch.randn(batch, head, seq, dim, requires_grad=True)
        k = torch.randn(batch, head, seq, dim, requires_grad=True)
        v = torch.randn(batch, head, seq, dim, requires_grad=True)
        mask = torch.ones(batch, 1, seq, seq)

        out_mha = mha(q, k, v, attention_mask=mask)
        out_ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask
        )
        self.assertEqual(out_mha, out_ref, atol=1e-3, rtol=1e-3)
