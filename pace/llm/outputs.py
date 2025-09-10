# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import Optional, List
from dataclasses import dataclass

import torch


@dataclass
class ModelOutput:
    """
    Output of the model. Contains the logits and key-value caches.
    """

    logits: torch.Tensor


@dataclass
class SamplerOutput:
    """
    Output of the sampler. Contains the sampled tokens, and
    optionally their probabilities and log probabilities.
    """

    next_tokens: torch.Tensor
    probs: Optional[torch.Tensor] = None
    logprobs: Optional[torch.Tensor] = None


@dataclass
class SpeculativeStats:
    """
    Statistics for speculative decoding. Contains the number of speculative
    tokens, the number of speculative tokens that were kept, and the number
    of speculative tokens that were discarded.
    """

    total_speculated_tokens: int
    mean_accepted_tokens: float


@dataclass
class GeneratorOutput:
    """
    Output of the generator. Contains the token ids of the generated text,
    and optionally the decoded text, input log probabilities, and
    output probabilities and log probabilities
    """

    output_token_ids: torch.Tensor
    decoded_text: Optional[List[str]] = None
    input_logprobs: Optional[torch.Tensor] = None
    probs: Optional[torch.Tensor] = None
    logprobs: Optional[torch.Tensor] = None
    speculative_stats: Optional[SpeculativeStats] = None
