# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import torch
from hypothesis import given
from hypothesis import strategies as st
from torch.testing._internal.common_utils import TestCase

from pace.llm import SamplingConfig
from pace.llm.stopping_criteria import StoppingCriteria
from pace.utils.logging import suppress_logging_cls


@suppress_logging_cls()
class TestStoppingCriteria(TestCase):

    @given(
        st.integers(min_value=1, max_value=100),
        st.sampled_from([torch.bfloat16, torch.float32]),
    )
    def test_stop_if_max_len(self, max_new_tokens, dtype):
        sampling_config = SamplingConfig(max_new_tokens=max_new_tokens)
        input_prompts = torch.rand((2, 5), dtype=dtype)
        stopping_criteria = StoppingCriteria(sampling_config, input_prompts)

        logits = torch.zeros((2, 5 + max_new_tokens), dtype=torch.long)
        self.assertTrue(stopping_criteria.stop_now(logits).max())

        logits = torch.zeros((2, 5), dtype=torch.long)
        self.assertFalse(stopping_criteria.stop_now(logits).max())

    @given(
        st.integers(min_value=1, max_value=1000),
        st.sampled_from([torch.bfloat16, torch.float32]),
    )
    def test_stop_if_eos_token(self, eos_token_id, dtype):
        sampling_config = SamplingConfig(eos_token_id=[eos_token_id])
        input_prompts = torch.rand((2, 5), dtype=dtype)
        stopping_criteria = StoppingCriteria(sampling_config, input_prompts)

        logits = torch.zeros((2, 6), dtype=torch.long)
        logits[0, -1] = eos_token_id
        self.assertTrue(stopping_criteria.stop_now(logits).max())

        logits[0, -1] = eos_token_id + 1
        self.assertFalse(stopping_criteria.stop_now(logits).max())

    @given(
        st.lists(st.integers(min_value=0, max_value=1000), min_size=1, max_size=5),
        st.sampled_from([torch.bfloat16, torch.float32]),
    )
    def test_stop_if_multiple_eos_tokens(self, eos_token_ids, dtype):
        sampling_config = SamplingConfig(eos_token_id=eos_token_ids)
        input_prompts = torch.rand((2, 5), dtype=dtype)
        stopping_criteria = StoppingCriteria(sampling_config, input_prompts)

        logits = torch.zeros((2, 6), dtype=torch.long)
        for eos_token_id in eos_token_ids:
            logits[0, -1] = eos_token_id
            self.assertTrue(stopping_criteria.stop_now(logits).max())
