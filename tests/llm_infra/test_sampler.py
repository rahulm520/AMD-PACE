# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import torch
from hypothesis import given
from hypothesis import strategies as st
from torch.testing._internal.common_utils import TestCase
import math

from pace.llm.configs import SamplingConfig
from pace.llm.sampler import Sampler
from pace.utils.logging import suppress_logging_cls

input_tensor = torch.randint(
    low=0,
    high=5,
    size=(2, 5),
    dtype=torch.int64,
)


@suppress_logging_cls()
class TestSampler(TestCase):
    def test_greedy_search(self):
        config = SamplingConfig(
            do_sample=False,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            min_p=0,
            num_beams=1,
            eos_token_id=[0],
        )
        config.verify_max_new_tokens()
        config.finalize()
        sampler = Sampler(config, input_tensor)
        logits = torch.randn(2, 5)
        output = sampler.sample(logits, input_tensor)
        self.assertEqual(output.next_tokens.shape, (2, 1))
        self.assertIsNotNone(output.probs)
        self.assertIsNotNone(output.logprobs)

    def test_random_sampling(self):
        config = SamplingConfig(
            do_sample=True,
            temperature=0.7,
            top_k=0,
            top_p=1.0,
            min_p=0,
            num_beams=1,
            eos_token_id=[0],
        )
        config.verify_max_new_tokens()
        config.finalize()
        sampler = Sampler(config, input_tensor)
        logits = torch.randn(2, 5)
        output = sampler.sample(input_tensor, logits)
        self.assertEqual(output.next_tokens.shape, (2, 1))
        self.assertIsNotNone(output.probs)
        self.assertIsNotNone(output.logprobs)

    def test_beam_search(self):
        num_beams = 2
        config = SamplingConfig(
            do_sample=False,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            min_p=0,
            num_beams=num_beams,
            eos_token_id=[0],
        )
        config.verify_max_new_tokens()
        config.finalize()
        sampler = Sampler(config, input_tensor)
        logits = torch.repeat_interleave(torch.randn(2, 5), num_beams, dim=0)
        beam_scores = torch.zeros(2 * num_beams)
        output = sampler.sample(input_tensor, logits, beam_scores)
        self.assertEqual(output.next_tokens.shape, (2, sampler.n_tokens_to_keep))
        self.assertIsNotNone(output.probs)

    @given(
        temperature=st.floats(min_value=0.01, max_value=5.0),
        batch_size=st.integers(min_value=1, max_value=1024),
        vocab_size=st.integers(min_value=2, max_value=1_000),
        min_p=st.floats(min_value=0.0, max_value=1.0),
        top_p=st.floats(min_value=0.0, max_value=1.0),
        top_k=st.integers(min_value=0, max_value=1_000),
    )
    def test_sampling_params(
        self, temperature, batch_size, vocab_size, min_p, top_p, top_k
    ):
        config = SamplingConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            num_beams=1,
            eos_token_id=[0],
        )
        config.verify_max_new_tokens()
        config.finalize()
        sampler = Sampler(config, input_tensor)
        logits = torch.randn(batch_size, vocab_size)
        output = sampler.sample(input_tensor, logits)
        self.assertEqual(output.next_tokens.shape, (batch_size, 1))
        self.assertIsNotNone(output.probs)
        self.assertIsNotNone(output.logprobs)

    def test_seed_reproducibility(self):
        config = SamplingConfig(
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            min_p=0,
            num_beams=1,
            eos_token_id=[0],
            sampling_seed=42,
        )
        config.verify_max_new_tokens()
        config.finalize()
        sampler1 = Sampler(config, input_tensor)
        logits = torch.randn(2, 5)
        output1 = sampler1.sample(input_tensor, logits)

        # Reset the seed in a new sampler
        sampler2 = Sampler(config, input_tensor)
        output2 = sampler2.sample(input_tensor, logits)

        self.assertTrue(torch.equal(output1.next_tokens, output2.next_tokens))

    def test_large_top_k_sampling(self):
        config = SamplingConfig(
            temperature=1.0,
            top_k=1_000_000,  # Larger than vocab
            top_p=1.0,
            min_p=0,
            num_beams=1,
            eos_token_id=[0],
        )
        config.verify_max_new_tokens()
        config.finalize()
        sampler = Sampler(config, input_tensor)
        logits = torch.randn(3, 4)
        output = sampler.sample(input_tensor, logits)
        self.assertEqual(output.next_tokens.shape, (3, 1))
        self.assertIsNotNone(output.probs)

    def test_zero_temperature(self):
        config = SamplingConfig(
            temperature=0.0000001,
            top_k=0,
            top_p=1.0,
            min_p=0,
            num_beams=1,
            eos_token_id=[0],
        )
        config.verify_max_new_tokens()
        config.finalize()
        sampler = Sampler(config, input_tensor)
        logits = torch.randn(3, 5)
        output = sampler.sample(input_tensor, logits)
        self.assertEqual(output.next_tokens.shape, (3, 1))
        self.assertIsNotNone(output.probs)
        self.assertIsNotNone(output.logprobs)

    def test_infinite_logits(self):
        config = SamplingConfig(
            temperature=1.0,
            top_k=2,
            top_p=1.0,
            min_p=0,
            num_beams=1,
            eos_token_id=[0],
        )
        config.verify_max_new_tokens()
        config.finalize()
        sampler = Sampler(config, input_tensor)
        logits = torch.randn(2, 5)
        logits[0, 3] = torch.nan
        with self.assertRaisesRegex(AssertionError, "Invalid logits"):
            sampler.sample(input_tensor, logits)

    def test_min_new_tokens(self):
        config = SamplingConfig(
            max_new_tokens=50,
            do_sample=True,
            temperature=0,
            top_k=50,
            random_seed=123,
            eos_token_id=1,
            min_new_tokens=4,
        )
        config.verify_max_new_tokens()
        config.finalize()
        sampler = Sampler(config, input_tensor)
        logits = torch.tensor(
            [[1.3, 0.41, -0.651, 0.1, 9.1], [1.21, 3.41, 0.71, 0.87, 0.1]]
        )
        output = sampler.sample(input_tensor, logits)
        self.assertEqual(output.logprobs[0][1], -math.inf)
        self.assertEqual(output.logprobs[1][1], -math.inf)

    def test_frequency_penalty_application(self):
        config = SamplingConfig(
            max_new_tokens=50,
            do_sample=True,
            temperature=0,
            random_seed=123,
            eos_token_id=3,
            frequency_penalty=1.87,
        )
        config.verify_max_new_tokens()
        config.finalize()
        sampler = Sampler(config, input_tensor)
        logits = torch.tensor(
            [[1.3, 0.22, 1.78, 12, 1.2], [-1.7, 1.43, 1.45, 0.651, 0.41]]
        )
        output = sampler.sample(input_tensor, logits)
        logit = torch.gather(logits, 1, input_tensor)
        logit = torch.where(
            logit < 0,
            logit * config.frequency_penalty,
            logit / config.frequency_penalty,
        )
        processed_logits = logits.scatter(1, input_tensor, logit)
        processed_logits = torch.softmax(processed_logits, dim=-1, dtype=torch.float)
        self.assertEqual(processed_logits, output.probs)

    def test_min_new_tokens_with_beam_search(self):
        config = SamplingConfig(
            max_new_tokens=50,
            do_sample=False,
            num_beams=2,
            temperature=0,
            eos_token_id=0,
            min_new_tokens=4,
        )
        config.verify_max_new_tokens()
        config.finalize()
        sampler = Sampler(config, input_tensor)
        logits = torch.repeat_interleave(torch.randn(2, 5), 2, dim=0)
        beam_scores = torch.zeros(2 * 2)
        output = sampler.sample(input_tensor, logits, beam_scores)
        check_eos_existence = torch.eq(output.next_tokens, 0).any().item()
        self.assertFalse(check_eos_existence)

    def test_frequency_penalty_application_with_beam_search(self):
        config = SamplingConfig(
            max_new_tokens=50,
            do_sample=False,
            temperature=1.0,
            num_beams=2,
            top_k=0,
            top_p=1.0,
            min_p=0,
            eos_token_id=[1],
            frequency_penalty=1.7,
        )
        config.finalize()
        sampler = Sampler(config, input_tensor)
        logits = torch.repeat_interleave(torch.randn(2, 5), 2, dim=0)
        beam_scores = torch.zeros(2 * 2)
        output = sampler.sample(input_tensor, logits, beam_scores)
        logit = torch.gather(logits, 1, input_tensor)
        logit = torch.where(
            logit < 0,
            logit * config.frequency_penalty,
            logit / config.frequency_penalty,
        )
        processed_logits = logits.scatter(1, input_tensor, logit)
        processed_logits = torch.log_softmax(
            processed_logits, dim=-1, dtype=torch.float
        )
        processed_logits = processed_logits + beam_scores[:, None].expand_as(
            processed_logits
        )
        vocab_size = processed_logits.shape[-1]
        processed_logits = processed_logits.view(-1, 2 * vocab_size)
        n_eos_tokens = (
            config.eos_token_id.shape[0] if config.eos_token_id is not None else 0
        )
        n_tokens_to_keep = max(2, 1 + n_eos_tokens) * config.num_beams
        processed_logits, next_tokens = torch.topk(
            processed_logits, n_tokens_to_keep, dim=1, largest=True, sorted=True
        )
        self.assertEqual(processed_logits, output.probs)
