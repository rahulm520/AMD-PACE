# *******************************************************************************
# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************

from functools import partial
from typing import Optional
import torch
from transformers import PreTrainedTokenizerBase, StopStringCriteria

from pace.llm.configs import SamplingConfig
from pace.utils.logging import PACE_LLM_ASSERT


# Stopping criteria for sampling
class StoppingCriteria(object):
    """
    Class to define stopping criteria for sampling. StoppingCriteria
    uses the SamplingConfig provided to define the stopping conditions.

    Args:
        sampling_config (SamplingConfig): The sampling configuration object.
    """

    def __init__(
        self,
        sampling_config: SamplingConfig,
        input_prompts: torch.Tensor,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):

        self.initial_decoder_input_length = input_prompts.shape[-1]

        self.stop_conditions = []  # To be used with conditions implemented in PACE
        self.hf_stop_conditions = []  # To be used with conditions implemented in HF
        if sampling_config.max_new_tokens is not None:
            max_length = (
                sampling_config.max_new_tokens + self.initial_decoder_input_length
            )
            self.stop_conditions.append(
                partial(self._stop_if_max_len, max_length=max_length)
            )

        if sampling_config.eos_token_id is not None:
            for eos_token_id in sampling_config.eos_token_id:
                self.stop_conditions.append(
                    partial(self._stop_if_eos_token, eos_token_id=eos_token_id)
                )

        if (
            sampling_config.stop_strings is not None
            and len(sampling_config.stop_strings) > 0
        ):
            if tokenizer is None:
                PACE_LLM_ASSERT(False, "Tokenizer is required for stop strings.")
            self.hf_stop_conditions.append(
                StopStringCriteria(
                    tokenizer=tokenizer, stop_strings=sampling_config.stop_strings
                )
            )

    def __str__(self):
        return (
            f"StoppingCriteria("
            f"stop_conditions={[(condition.func.__name__, condition.keywords) for condition in self.stop_conditions]}, "
            f"hf_stop_conditions={self.hf_stop_conditions})"
        )

    def __repr__(self):
        return str(self)

    def _stop_if_max_len(self, logits: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Check if the number of new tokens generated is greater than max_new_tokens.

        Args:
            logits (torch.Tensor): The logits tensor.
            max_new_tokens (int): The maximum number of new tokens allowed.

        Returns:
            torch.Tensor: A tensor of boolean values indicating if the length
                of the generated sequence is greater than max_new_tokens.
        """
        cur_len = logits.shape[-1]
        is_done = cur_len >= max_length
        return torch.full((logits.shape[0],), is_done, dtype=torch.bool)

    def _stop_if_eos_token(
        self, logits: torch.Tensor, eos_token_id: int
    ) -> torch.Tensor:
        """
        Check if the last token generated is the EOS token.

        Args:
            logits (torch.Tensor): The logits tensor.
            eos_token_id (int): The EOS token id.

        Returns:
            torch.Tensor: A tensor of boolean values indicating if the last
                token generated is the EOS token.
        """
        return torch.isin(logits[:, -1], eos_token_id)

    def _check_for_conditions(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Check if any of the stop conditions are met.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            torch.Tensor: A tensor of boolean values indicating if the sampling
                should be stopped.
        """

        # Initialize is_done to False, with one value for each sample
        # in the batch. If any of the stop conditions are met, the
        # corresponding value in is_done is set to True.
        is_done = torch.full((logits.shape[0],), False, dtype=torch.bool)
        for condition in self.stop_conditions:
            is_done = is_done | condition(logits=logits)

        if self.hf_stop_conditions:
            for condition in self.hf_stop_conditions:
                is_done = is_done | condition(logits, scores=None)
        return is_done

    def stop_now(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Check if the sampling should be stopped. If any of the stop conditions
        are met, the corresponding value in the output tensor is set to True.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            torch.Tensor: A tensor of boolean values indicating if the sampling
                should be stopped.
        """

        return self._check_for_conditions(logits)
