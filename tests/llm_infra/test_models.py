# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import torch
from hypothesis import given
from hypothesis import strategies as st
from pace.utils.logging import suppress_logging_cls
from torch.testing._internal.common_utils import TestCase
from pace.llm.models.model_utils import set_default_torch_dtype


@suppress_logging_cls()
class TestModelUtils(TestCase):

    @given(dtype=st.sampled_from([torch.float32, torch.bfloat16]))
    def test_set_default_torch_dtype(self, dtype):
        old_dtype = torch.get_default_dtype()
        with set_default_torch_dtype(dtype):
            self.assertEqual(torch.get_default_dtype(), dtype)
        self.assertEqual(torch.get_default_dtype(), old_dtype)
