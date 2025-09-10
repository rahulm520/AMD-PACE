# *******************************************************************************
# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************

from functools import partial
from typing import Optional

import torch
import math

from pace.llm.configs import SamplingConfig, SamplingMode
from pace.llm.outputs import SamplerOutput
from pace.utils.logging import PACE_LLM_ASSERT


class Sampler(object):
    """
    Sampler class to sample from the model's output logits. The sampler can be configured to use
    different sampling strategies like Beam Search, Greedy Search, Random Sampling, Top-k, Top-p, Min-p sampling.

    Args:
        sampling_config: SamplingConfig object with sampling configuration.
    """

    def __init__(
        self,
        sampling_config: SamplingConfig,
        input_encodings: torch.Tensor,
    ) -> None:
        self.sampling_config = sampling_config
        self._set_sampling_seed(sampling_config.sampling_seed)
        self.penalty = self.sampling_config.frequency_penalty
        self.initial_input_length = input_encodings.shape[-1]
        self.sampler_preprocessors = []
        if sampling_config.top_k != 0:
            self.sampler_preprocessors.append(
                partial(Sampler._apply_top_k, top_k=sampling_config.top_k)
            )
        if sampling_config.top_p < 1.0:
            self.sampler_preprocessors.append(
                partial(Sampler._apply_tok_p, top_p=sampling_config.top_p)
            )
        if sampling_config.min_p > 0:
            self.sampler_preprocessors.append(
                partial(Sampler._apply_min_p, min_p=sampling_config.min_p)
            )

        self.do_sampler_preprocessors = False
        if len(self.sampler_preprocessors) > 0:
            self.do_sampler_preprocessors = True

        # n_tokens_to_keep is the number of tokens to keep in the beam search,
        # to be provided to top_k: https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/generation/utils.py#L3541
        n_eos_tokens = (
            self.sampling_config.eos_token_id.shape[0]
            if self.sampling_config.eos_token_id is not None
            else 0
        )
        self.n_tokens_to_keep = (
            max(2, 1 + n_eos_tokens) * self.sampling_config.num_beams
        )

    def _set_sampling_seed(self, sampling_seed: Optional[int] = None) -> None:
        """
        Set the seed for sampling operations. This is useful for reproducibility.

        Args:
            sampling_seed: Seed for sampling operations
        """
        if sampling_seed is not None:
            torch.manual_seed(sampling_seed)

    # Top-k, Top-p, Min-p samplers adapted from Huggingface's transformers library
    # https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/generation/logits_process.py
    @staticmethod
    def _apply_top_k(logits: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
        top_k = min(top_k, logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits_processed = logits.masked_fill(
            indices_to_remove, torch.finfo(logits.dtype).min
        )
        return logits_processed

    @staticmethod
    def _apply_tok_p(logits: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -1:] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits_processed = logits.masked_fill(
            indices_to_remove, torch.finfo(logits.dtype).min
        )
        return logits_processed

    @staticmethod
    def _apply_min_p(logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)
        # Get the probability of the top token for each sequence in the batch
        top_probs, _ = probs.max(dim=-1, keepdim=True)
        # Calculate the actual min_p threshold by scaling min_p with the top token's probability
        scaled_min_p = min_p * top_probs
        # Create a mask for tokens that have a probability less than the scaled min_p
        tokens_to_remove = probs < scaled_min_p

        sorted_indices = torch.argsort(logits, descending=True, dim=-1)
        sorted_indices_to_remove = torch.gather(
            tokens_to_remove, dim=-1, index=sorted_indices
        )
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., :1] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits_processed = logits.masked_fill(
            indices_to_remove, torch.finfo(logits.dtype).min
        )
        return logits_processed

    def apply_repetition_penalty(
        self, logits: torch.Tensor, input_encodings
    ) -> torch.Tensor:
        """
        A penalty above 1.0 increases the odds of selecting tokens that were present in the prompt. A penalty between 0 and 1.0
        increases the odds of selecting tokens that were not present in the prompt. This is useful to reduce hallucinations.
        Args:
            logits: Model's output logits
            input_encodings: Encoded input representations.
        Returns:
            logits: Processed logits with repetition penalty applied

        """
        logit = torch.gather(logits, 1, input_encodings)
        # if score < 0 then hallucination penalty has to be multiplied to increase the token probabilities
        logit = torch.where(logit < 0, logit * self.penalty, logit / self.penalty)
        logits_processed = logits.scatter(1, input_encodings, logit)
        return logits_processed

    def set_min_new_tokens(self, logits: torch.Tensor, input_encodings) -> torch.Tensor:
        """
        enforcing a min-length of new tokens by setting EOS (End-Of-Sequence) token probability to 0.
        Args:
            logits: Model's output logits
            input_encodings: Encoded input representations.
        Returns:
            logits: Processed logits with EOS token probability set to 0
        """
        eos_token_id = self.sampling_config.eos_token_id
        vocab_tensor = torch.arange(logits.shape[-1], device=logits.device)
        eos_token_mask = torch.isin(vocab_tensor, eos_token_id)
        scores_processed = logits.clone()
        if input_encodings.shape[-1] < (
            self.initial_input_length + self.sampling_config.min_new_tokens
        ):
            scores_processed = torch.where(eos_token_mask, -math.inf, logits)
        return scores_processed

    def _beam_search(
        self,
        logits: torch.Tensor,
        prev_beam_scores: torch.Tensor,
        input_encodings: torch.Tensor,
    ) -> SamplerOutput:
        """
        Beam search implementation to sample the next token from the model's output logits.

        Args:
            logits: Model's output logits
            prev_beam_scores: Beam scores from the previous step
            input_encodings: Encoded input representations.

        Returns:
            SamplerOutput object with the sampled token and its probabilities
        """
        if self.penalty != 1.0:
            logits = self.apply_repetition_penalty(logits, input_encodings)
        if self.sampling_config.min_new_tokens > 0:
            logits = self.set_min_new_tokens(logits, input_encodings)
        next_token_scores = torch.log_softmax(logits, dim=-1)
        next_token_scores_processed = next_token_scores
        if self.do_sampler_preprocessors:
            for sampler_preprocessor in self.sampler_preprocessors:
                next_token_scores_processed = sampler_preprocessor(
                    logits=next_token_scores_processed
                )
        next_token_scores = next_token_scores_processed + prev_beam_scores[
            :, None
        ].expand_as(next_token_scores_processed)
        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(
            -1, self.sampling_config.num_beams * vocab_size
        )

        next_token_scores, next_tokens = torch.topk(
            next_token_scores, self.n_tokens_to_keep, dim=1, largest=True, sorted=True
        )

        # Both probs and logprobs are None for now
        return SamplerOutput(next_tokens, next_token_scores, None)

    def _sample(
        self, logits: torch.Tensor, input_encodings: torch.Tensor
    ) -> SamplerOutput:
        """
        Sample the next token from the model's output logits using the configured sampling strategy
        for random sampling and greedy search.

        Args:
            logits: Model's output logits
            input_encodings: Encoded input representations.

        Returns:
            SamplerOutput object with the sampled token, probabilities, and log probabilities
        """
        # Apply temperature scaling.
        # Use in-place division to avoid creating a new tensor.
        if self.sampling_config.temperature != 1.0:
            logits.div_(self.sampling_config.temperature)
        if self.penalty != 1.0:
            logits = self.apply_repetition_penalty(logits, input_encodings)
        if self.sampling_config.min_new_tokens > 0:
            logits = self.set_min_new_tokens(logits, input_encodings)
        # Apply top_k, top_p, min_p
        preprocessed_logits = logits
        if self.do_sampler_preprocessors:
            for sampler_preprocessor in self.sampler_preprocessors:
                preprocessed_logits = sampler_preprocessor(logits=preprocessed_logits)

        # Compute the probabilities.
        probs = torch.softmax(preprocessed_logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        logprobs = torch.log_softmax(preprocessed_logits, dim=-1, dtype=torch.float)

        next_tokens = None
        if self.sampling_config.sampling_mode == SamplingMode.GREEDY_SEARCH:
            next_tokens = torch.argmax(probs, dim=-1, keepdim=True)
        else:  # By default, use random sampling
            next_tokens = torch.multinomial(probs, num_samples=1)

        # Make sure the next token is valid.
        PACE_LLM_ASSERT(
            next_tokens is not None or torch.isnan(next_tokens).any(),
            "Invalid next token sampled, something went wrong!",
        )

        return SamplerOutput(next_tokens, probs, logprobs)

    def sample(
        self,
        input_encodings: torch.Tensor,
        logits: torch.Tensor,
        beam_scores: Optional[torch.Tensor] = None,
    ) -> SamplerOutput:
        """
        Sample the next token from the model's output logits using the configured sampling strategy.

        Args:
            input_encodings: Encoded input representations.
            logits: Model's output logits
            beam_scores: Cumulative log probabilities per beam.

        Returns:
            ModelOutput object with the sampled token, probabilities, and log probabilities
        """

        PACE_LLM_ASSERT(
            logits is not None and not torch.isnan(logits).any(),
            "Invalid logits provided for sampling, something went wrong!",
        )

        if self.sampling_config.sampling_mode == SamplingMode.BEAM_SEARCH:
            return self._beam_search(logits, beam_scores, input_encodings)
        else:
            return self._sample(logits, input_encodings)
