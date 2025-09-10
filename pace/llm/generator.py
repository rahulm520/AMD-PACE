# *******************************************************************************
# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************

import os
from typing import Union, Optional, Tuple

import torch
from transformers import (
    PreTrainedTokenizer,
    PretrainedConfig,
    BatchEncoding,
    BeamSearchScorer,
    TextStreamer,
)
from transformers.utils import CONFIG_NAME, GENERATION_CONFIG_NAME

from pace.llm.sampler import Sampler
from pace.llm.configs import (
    SamplingConfig,
    SamplingMode,
    OperatorConfig,
    PardSpecDecodeConfig,
)
from pace.llm.cache import KVCacheType, KVCacheManager
from pace.llm.stopping_criteria import StoppingCriteria
from pace.llm.models.hf_utils import resolve_model_path
from pace.llm.models.model_utils import init_model, get_tokenizer
from pace.llm.outputs import (
    ModelOutput,
    SamplerOutput,
    GeneratorOutput,
    SpeculativeStats,
)
from pace.utils.logging import PACE_LLM_DEBUG, PACE_LLM_INFO, PACE_LLM_ASSERT


def validate_generator_inputs(
    model_path: Union[str, os.PathLike],
    tokenizer_path: Optional[Union[str, os.PathLike]] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16,
) -> None:
    """
    Validates the inputs for the Generator class

    Args:
        model_path (Union[str, os.PathLike]): Path to the model
        tokenizer_path (Optional[Union[str, os.PathLike]]): Path to the tokenizer
        dtype (Optional[torch.dtype]): Data type for the model

    Raises:
        FileNotFoundError: If the model or tokenizer path does not exist
        TypeError: If the dtype is not a torch.dtype
    """

    def validate_path(path: os.PathLike, error_message: str):
        if not os.path.exists(path):
            raise FileNotFoundError(error_message)

    validate_path(
        model_path,
        f"The model path provided does not exist. Please receck path: {model_path}",
    )

    config_path = os.path.join(model_path, CONFIG_NAME)
    validate_path(
        config_path,
        f"The model path provided does not have {CONFIG_NAME}. Please recheck path: , {model_path}",
    )

    if tokenizer_path is not None:
        validate_path(
            tokenizer_path,
            f"The tokenizer path provided does not exist. Please recheck path: {tokenizer_path}",
        )

    if not (isinstance(dtype, torch.dtype)):
        raise TypeError(
            f"Generator input dtype should be a torch.dtype, got {type(dtype)}"
        )


class Generator(object):
    """
    A class to generate text from a given model. This is the backend to
    the LLMModel class and is to be used internally. The class is initialized
    with a model, an optional tokenizer path, and an optional data type.

    Tokenizer is requried for the model to work. If the tokenizer is not provided,
    the model will try to load the tokenizer from the model path. If the tokenizer
    is not present in the model path, an error will be raised.

    Args:
        model_name_or_path (Union[str, os.PathLike]): Path to the model or the model name
        tokenizer_name_or_path (Optional[Union[str, os.PathLike]]):Path to the tokenizer or tokenizer name, if any
        dtype (Optional[torch.dtype]): Data type for the model, defaults to torch.bfloat16

    Raises:
        FileNotFoundError: If the model or tokenizer path does not exist
        TypeError: If the dtype is not a torch.dtype
    """

    def __init__(
        self,
        model_name_or_path: Union[str, os.PathLike],
        tokenizer_name_or_path: Optional[Union[str, os.PathLike]] = None,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        kv_cache_type: Optional[KVCacheType] = KVCacheType.DYNAMIC,
        pard_config: Union[PardSpecDecodeConfig] = None,
        opconfig: Optional[OperatorConfig] = None,
        disable_tqdm: Optional[bool] = False,
    ):

        self.model_path = resolve_model_path(model_name_or_path)
        self.tokenizer_path = None
        if tokenizer_name_or_path is not None:
            self.tokenizer_path = resolve_model_path(tokenizer_name_or_path)
        validate_generator_inputs(self.model_path, self.tokenizer_path, dtype)

        self.pard_config = None
        if pard_config is not None:
            self.pard_config = pard_config
            self.pard_model_path = resolve_model_path(pard_config.model_name_or_path)
            validate_generator_inputs(self.pard_model_path, self.tokenizer_path, dtype)

        opconfig = (
            opconfig.finalize() if opconfig is not None else OperatorConfig().finalize()
        )
        self.model = init_model(
            self.model_path, dtype=dtype, opconfig=opconfig, disable_tqdm=disable_tqdm
        )
        self.tokenizer = get_tokenizer(self.model_path, self.tokenizer_path)
        if self.pard_config and self.pard_model_path:
            self.pard_model = init_model(
                self.pard_model_path, dtype=dtype, opconfig=opconfig
            )
        self.kv_cache_type = kv_cache_type

    def _prepare_inputs(
        self, prompts: Union[torch.Tensor, BatchEncoding]
    ) -> torch.Tensor:
        """
        Prepares the inputs for the model. If the inputs are a BatchEncoding, extracts the input_ids
        and returns it. If the inputs are a tensor, returns the tensor as is.

        Args:
            prompts (Union[torch.Tensor, BatchEncoding]): The input prompts

        Returns:
            torch.Tensor: The input prompts
        """

        if isinstance(prompts, BatchEncoding):
            input_prompts = prompts.input_ids
        else:
            input_prompts = prompts

        return input_prompts

    def _prepare_sampling_config(
        self,  # type: ignore
        user_sampling_config: Optional[SamplingConfig] = None,
        initial_decoder_input_length: Optional[int] = 0,
        model_max_new_tokens: Optional[int] = 2048,
    ) -> SamplingConfig:
        """
        Prepares the sampler for the model. If the user sampling config is present, merges it with the model's
        sampling config. If the user sampling config is not present, uses the model's sampling config.

        Args:
            user_sampling_config (Optional[SamplingConfig]): The user sampling config
            initial_decoder_input_length: Original length of inputs.
            model_max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            SamplingConfig: The sampling config
        """

        # Create an empty sampling config
        sampling_config = SamplingConfig()

        # If the generation config is present in the model, load it
        generation_config_from_model = os.path.join(
            self.model_path, GENERATION_CONFIG_NAME
        )
        if os.path.exists(generation_config_from_model):
            sampling_config = SamplingConfig.from_pretrained(
                generation_config_from_model
            )

        # Merge the user sampling config with the model's
        sampling_config.merge_from(user_sampling_config, self.tokenizer)
        sampling_config.verify_max_new_tokens(
            initial_decoder_input_length, model_max_new_tokens
        )
        sampling_config.finalize()

        return sampling_config

    def _create_attention_mask(
        self,
        input_prompts: Union[torch.Tensor, BatchEncoding],
        sampling_config: SamplingConfig,
    ):
        """
        Creates the attention mask for the model. If the inputs are a BatchEncoding, extracts the attention mask
        and returns it. If the inputs are a tensor, creates the attention mask based on the pad_token_id.

        Adapted from HuggingFace's implementation:
        https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/generation/utils.py#L476

        Args:
            input_prompts (Union[torch.Tensor, BatchEncoding]): The input prompts
            sampling_config (SamplingConfig): The sampling config

        Returns:
            torch.Tensor: The attention mask
        """

        # If the inputs is a BatchEncoding, extract the attention mask and return it
        if isinstance(input_prompts, BatchEncoding):
            return input_prompts.attention_mask

        # Only work with tensors from now on
        PACE_LLM_ASSERT(
            isinstance(input_prompts, torch.Tensor),
            f"Input prompts should be a tensor or a BatchEncoding, got {type(input_prompts)}",
        )

        default_attention_mask = torch.ones(input_prompts.shape[:2], dtype=torch.long)
        if sampling_config.pad_token_id is None:
            return default_attention_mask

        is_input_ids = len(input_prompts.shape) == 2 and input_prompts.dtype in [
            torch.int,
            torch.long,
        ]
        if not is_input_ids:
            return default_attention_mask

        is_pad_token_in_inputs = (sampling_config.pad_token_id is not None) and (
            torch.isin(
                elements=input_prompts, test_elements=sampling_config.pad_token_id
            ).any()
        )
        is_pad_token_not_equal_to_eos_token_id = (
            sampling_config.eos_token_id is None
        ) or ~(
            torch.isin(
                elements=sampling_config.eos_token_id,
                test_elements=sampling_config.pad_token_id,
            ).any()
        )
        can_infer_attention_mask = (
            is_pad_token_in_inputs * is_pad_token_not_equal_to_eos_token_id
        )
        attention_mask_from_padding = input_prompts.ne(
            sampling_config.pad_token_id
        ).long()

        attention_mask = (
            attention_mask_from_padding * can_infer_attention_mask
            + default_attention_mask * ~can_infer_attention_mask
        )
        return attention_mask

    def _prepare_streamer(
        self, text_streamer: TextStreamer, input_prompts: torch.Tensor
    ) -> Optional[TextStreamer]:

        if text_streamer:
            PACE_LLM_ASSERT(
                self.sampling_config.sampling_mode != SamplingMode.BEAM_SEARCH,
                "Text streamer is not supported for beam search",
            )
            if isinstance(text_streamer, TextStreamer):
                PACE_LLM_ASSERT(
                    input_prompts.shape[0] == 1,
                    "Text streamer is only supported for batch size of 1",
                )
            return text_streamer
        return None

    def _adjust_mask_for_generation(
        self,
        attention_mask: torch.Tensor,
        unfinished_sequences: Optional[torch.Tensor] = None,
        draft_size: Optional[int] = 1,
    ) -> torch.Tensor:
        """
        Adjusts the attention mask based on the stopping criteria output

        Args:
            attention_mask (torch.Tensor): The attention mask
            unfinished_sequences (torch.Tensor): A vector containing 1s for unfinished sequences and 0s for finished sequences

        Returns:
            torch.Tensor: The adjusted attention mask
        """

        attention_mask = torch.cat(
            [
                attention_mask,
                attention_mask.new_ones((attention_mask.shape[0], draft_size)),
            ],
            dim=-1,
        )
        # If a sequence is finished, the attention mask for the next token should be 0
        if unfinished_sequences is not None:
            attention_mask = attention_mask * unfinished_sequences.unsqueeze(-1)
        return attention_mask

    def _update_probs_logprobs(
        self,
        probs: torch.Tensor,
        logprobs: torch.Tensor,
        sampler_output: SamplerOutput,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the probs and logprobs based on the sampler output.

        NOTE: This is not supported for beam search since it will have different probabilities
        accross different beams and it will be difficult to track the probabilities.

        Args:
            probs (torch.Tensor): The probabilities
            logprobs (torch.Tensor): The log probabilities
            sampler_output (SamplerOutput): The sampler output

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The updated probabilities and log probabilities
        """

        if self.sampling_config.return_probs:
            if probs is None:
                probs = sampler_output.probs
            else:
                probs = torch.cat([probs, sampler_output.probs], dim=-1)

        if self.sampling_config.return_logprobs:
            if logprobs is None:
                logprobs = sampler_output.logprobs
            else:
                logprobs = torch.cat([logprobs, sampler_output.logprobs], dim=-1)

        return probs, logprobs

    def _prepare_output_for_generate(
        self,
        output_token_ids: torch.Tensor,
        input_logprobs: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        logprobs: Optional[torch.Tensor] = None,
        decoded_text: Optional[str] = None,
    ) -> GeneratorOutput:
        """
        Prepares the output for the generate method in the GeneratorOutput format

        Args:
            output_token_ids (torch.Tensor): The output token ids
            logprobs (torch.Tensor): The log probabilities
            input_logprobs (torch.Tensor): The input log probabilities
            decoded_text (Optional[str]): The decoded text

        Returns:
            GeneratorOutput: The generated outputs
        """

        decoded_text = None
        if self.sampling_config.return_text:
            decoded_text = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in output_token_ids
            ]

        speculative_stats = None
        if self.pard_config:
            speculative_stats = SpeculativeStats(
                total_speculated_tokens=sum(self.total_speculated_tokens),
                mean_accepted_tokens=sum(self.total_speculated_tokens)
                / len(self.total_speculated_tokens),
            )

        output = GeneratorOutput(
            output_token_ids=output_token_ids,
            probs=probs,
            logprobs=logprobs,
            input_logprobs=input_logprobs,
            decoded_text=decoded_text,
            speculative_stats=speculative_stats,
        )
        return output

    def _prepare_kv_cache(
        self, kv_cache_type: KVCacheType, input_prompts: torch.Tensor
    ):
        """
        Prepares the key-value cache for the model. If the kv_cache_type is not None, initializes the key-value cache
        based on the type.

        Args:
            kv_cache_type (KVCacheType): The key-value cache type
        """

        max_seq_length = self.sampling_config.max_new_tokens + input_prompts.size(-1)
        self.kv_cache_manager = KVCacheManager(
            self.model.config, max_seq_length, kv_cache_type
        )

    def _prepare_for_pard(
        self,
        input_prompts: torch.Tensor,
    ):

        if self.pard_config is None:
            self.draft_size = 0
            return

        PACE_LLM_ASSERT(
            self.sampling_config.sampling_mode == SamplingMode.GREEDY_SEARCH,
            "Speculative Decoding using PARD is only supported for greedy "
            f"search sampling mode but got {self.sampling_config.sampling_mode}",
        )

        PACE_LLM_ASSERT(
            input_prompts.shape[0] == 1,
            "Speculative Decoding using PARD is only supported for batch size of 1",
        )

        # Overwrite the pard_token with the one from the model config if it exists
        self.pard_config.pard_token = getattr(
            self.pard_model.config, "pard_token", self.pard_config.pard_token
        )
        self.pard_token_list = [self.pard_config.pard_token for i in range(32)]
        self.draft_size = self.pard_config.num_speculative_tokens

        self.total_speculated_tokens = []

        self.pard_kv_cache_manager = KVCacheManager(
            self.pard_model.config,
            self.sampling_config.max_new_tokens + input_prompts.size(-1),
            self.kv_cache_type,
        )

        PACE_LLM_INFO(
            f"Using PARD for speculative decoding, using config: {self.pard_config}"
        )

    def prepare_for_generate(
        self,
        prompts: Union[torch.Tensor, BatchEncoding],
        sampling_config: Optional[SamplingConfig] = None,
        text_streamer: Optional[TextStreamer] = None,
    ) -> torch.Tensor:
        """
        Prepares the inputs for the model, prepares the sampler and creates the attention mask.

        Args:
            prompts (Union[torch.Tensor, BatchEncoding]): The input prompts
            sampling_config (Optional[SamplingConfig]): The sampling config

        Returns:
            torch.Tensor: The input prompts

        Raises:
            NotImplementedError: If the input prompts or sampling config is not a torch.Tensor or BatchEncoding
            NotImplementedError: If the sampling config is not a SamplingConfig
        """

        if not isinstance(prompts, (torch.Tensor, BatchEncoding)):
            raise NotImplementedError(
                f"Only input types of torch.Tensor or BatchEncoding is allowed for now, got {type(prompts)}"
            )

        if sampling_config is not None and not isinstance(
            sampling_config, SamplingConfig
        ):
            raise NotImplementedError(
                f"Only input types of SamplingConfig is allowed for now, got {type(sampling_config)}"
            )

        # Converts everything into tensors
        input_prompts = self._prepare_inputs(prompts)
        initial_decoder_input_length = input_prompts.shape[-1]

        # Prepare configs and intantiate the sampler
        self.sampling_config = self._prepare_sampling_config(
            sampling_config,
            initial_decoder_input_length,
            self.model.config.max_position_embeddings,
        )
        self.sampler = Sampler(self.sampling_config, input_prompts)

        # Creates attention mask to start with
        self.attention_mask = self._create_attention_mask(prompts, self.sampling_config)

        # Prepare key-value cache
        self._prepare_kv_cache(self.kv_cache_type, input_prompts)

        # Creates stoppping config
        self.stopping_criteria = StoppingCriteria(
            self.sampling_config, input_prompts, self.tokenizer
        )

        self.text_streamer = self._prepare_streamer(text_streamer, input_prompts)

        # If PARD is configured, prepare to use
        self._prepare_for_pard(input_prompts)

        PACE_LLM_INFO(str(self.sampling_config))
        PACE_LLM_INFO(str(self.stopping_criteria))

        return input_prompts

    def _beam_search(self, inputs: torch.Tensor) -> GeneratorOutput:
        """
        Beam search implementation.
            Beam search seems to be an outdated technique, and
            most of the models does not seem to be using it.
            However, it is still available in the transformers library,
            and some of the priority models require beam search,
            thus it is implemented here.
            Might not be the most efficient implementation, and might
            not be performant for large models.

        Args:
            inputs (torch.Tensor): The input prompts

        Returns:
            GeneratorOutput: The generated outputs
        """

        # Calculate info required by beam search
        batch_size = inputs.shape[0]
        initial_decoder_input_length = inputs.shape[-1]
        max_tokens = initial_decoder_input_length + self.sampling_config.max_new_tokens

        # Empty beam indices and scores
        beam_indices = None
        beam_scores = torch.zeros(
            (batch_size, self.sampling_config.num_beams),
            dtype=torch.float,
        )
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * self.sampling_config.num_beams,))

        # Initialize the beam scorer
        beam_scorer = BeamSearchScorer(
            batch_size=inputs.shape[0],
            num_beams=self.sampling_config.num_beams,
            device=torch.device("cpu"),
            do_early_stopping=self.sampling_config.early_stopping,
            max_length=max_tokens,
        )

        # Inputs and masks are broadcasted to the beam size
        inputs = inputs.repeat_interleave(self.sampling_config.num_beams, dim=0)
        self.attention_mask = self.attention_mask.repeat_interleave(
            self.sampling_config.num_beams, dim=0
        )

        # Unfinished sequences are initialized to all ones, so that means
        # all sequences are unfinished, and once a sequence is finished, it will
        # be set to zero in the corresponding index, and the loop will continue
        # until all sequences are finished.
        unfinished_sequences = torch.ones(inputs.shape[0], dtype=torch.long)
        beam_search_stop = False

        while unfinished_sequences.max() != 0 and (not beam_search_stop):
            model_out: ModelOutput = self.model(
                inputs,
                kv_cache=self.kv_cache_manager,
                attention_mask=self.attention_mask,
            )
            logits = model_out.logits

            # Sample the next token using the sampler
            next_token_logits = logits[:, -1, :].clone().float()
            sampler_output: SamplerOutput = self.sampler.sample(
                inputs, next_token_logits, beam_scores
            )

            next_tokens = sampler_output.next_tokens
            vocab_size = next_token_logits.shape[-1]

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size
            beam_outputs = beam_scorer.process(
                inputs,
                sampler_output.probs,
                next_tokens,
                next_indices,
                pad_token_id=self.sampling_config.pad_token_id,
                eos_token_id=self.sampling_config.eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=initial_decoder_input_length,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]
            beam_search_stop = beam_scorer.is_done

            inputs = torch.cat(
                [inputs[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
            )
            # Use KVCacheManager to update the KV cache
            self.kv_cache_manager.update_cache_with_beam_indices(beam_idx)

            # Check if the sequence is finished, using the stopping criteria
            stopping_criteria_output = self.stopping_criteria.stop_now(inputs)
            unfinished_sequences = unfinished_sequences & ~stopping_criteria_output

            # Update the attention mask
            self.attention_mask = self._adjust_mask_for_generation(
                self.attention_mask, unfinished_sequences
            )

        # Finalize the beam search and remove the beam dimension
        sequence_outputs = beam_scorer.finalize(
            inputs,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=self.sampling_config.pad_token_id,
            eos_token_id=self.sampling_config.eos_token_id,
            max_length=max_tokens,
            beam_indices=beam_indices,
            decoder_prompt_len=initial_decoder_input_length,
        )

        return self._prepare_output_for_generate(
            output_token_ids=sequence_outputs["sequences"]
        )

    def _speculate_tokens(
        self, input_prompts: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # Short circuit if PARD is not configured
        if self.pard_config is None:
            return input_prompts

        # Append the PARD token to the input prompts
        unused_tokenids = torch.tensor([self.pard_token_list])[:, : self.draft_size - 1]
        pard_input_prompts = torch.cat([input_prompts, unused_tokenids], dim=-1)
        # Adjust attention mask for PARD
        attention_mask = self._adjust_mask_for_generation(
            attention_mask, draft_size=self.draft_size - 1
        )

        speculated_output: ModelOutput = self.pard_model(
            pard_input_prompts,
            kv_cache=self.pard_kv_cache_manager,
            attention_mask=attention_mask,
        )
        speculated_tokens = speculated_output.logits[:, -self.draft_size :].argmax(
            dim=-1
        )

        # Add one for the extra generated token
        self.attention_mask = self._adjust_mask_for_generation(
            attention_mask, draft_size=1
        )

        return torch.cat([input_prompts, speculated_tokens], dim=-1)

    def _postprocess_speculated_tokens(
        self,
        next_tokens: torch.Tensor,
        speculated_tokens: torch.Tensor,
    ) -> torch.Tensor:

        # short circuit if PARD is not configured
        if self.pard_config is None:
            return next_tokens

        speculated_tokens = speculated_tokens[:, -self.draft_size :]
        keep_token_ids = []

        # Only check the first draft_size - 1 tokens since the last token is the
        # one generated by the target model which is always kept
        for i in range(self.draft_size - 1):
            # Hardcoded to 0 since we only have one batch
            if next_tokens[0][i] == speculated_tokens[0][i]:
                keep_token_ids.append(next_tokens[:, i])
            else:
                break

        # Add the token generated by the target model
        keep_token_ids.append(next_tokens[:, len(keep_token_ids)])
        keep_token_ids = torch.concat(keep_token_ids).reshape(1, -1)

        # Remove the KV Cache for the tokens that are not kept
        remove_token_id_len = self.draft_size - keep_token_ids.shape[1]
        # If remove tokens are 0, then we don't need to remove anything
        # since the model has already updated the KV Cache
        if remove_token_id_len > 0:
            # +1 to account for the token which should be used for
            # the next iteration
            self.kv_cache_manager.remove_cache(remove_token_id_len + 1)
        # Remove all the KV Cache for PARD tokens from the current iteration
        # so that it recomputes the KV Cache with the accepted tokens
        # in the next iteration
        self.pard_kv_cache_manager.remove_cache(self.draft_size - 1)

        # Remove all the extra padding from attention mask, the number of
        # tokens accepted will be updated in a later stage
        self.attention_mask = self.attention_mask[:, : -self.draft_size]

        self.total_speculated_tokens.append(keep_token_ids.shape[1])
        return keep_token_ids

    def _generate(self, inputs: torch.Tensor) -> GeneratorOutput:
        """
        Generates text from the model given the input prompts.
        Implemented for sampling modes other than beam search (greedy and random sampling)

        Args:
            inputs (torch.Tensor): The input prompts

        Returns:
            GeneratorOutput: The generated outputs
        """

        # @TODO: Needs rework in the future
        # All the initilizations for outputs
        input_logprobs = None
        probs = None
        logprobs = None

        if self.text_streamer:
            self.text_streamer.put(inputs)

        # Unfinished sequences are initialized to all ones, so that means
        # all sequences are unfinished, and once a sequence is finished, it will
        # be set to zero in the corresponding index, and the loop will continue
        # until all sequences are finished.
        unfinished_sequences = torch.ones(inputs.shape[0], dtype=torch.long)
        while unfinished_sequences.max() != 0:
            # Run the inputs through the model

            speculated_inputs = self._speculate_tokens(inputs, self.attention_mask)
            model_out: ModelOutput = self.model(
                speculated_inputs,
                kv_cache=self.kv_cache_manager,
                attention_mask=self.attention_mask,
            )

            logits = model_out.logits

            # One time calculation of input logprobs
            if input_logprobs is None and self.sampling_config.return_input_logprobs:
                input_logprobs = torch.log_softmax(logits, dim=-1)

            # Sample the next token using the sampler
            # next_token_logits = logits[:, -self.draft_size - 1 :, :].clone().float()
            next_token_logits = logits[:, -self.draft_size - 1 :, :].clone().float()
            # Squeeze the logits if the draft size is 0
            if next_token_logits.size(1) == 1 and self.draft_size == 0:
                # This is the case when we are not speculating any tokens, and we just want to get the next token
                # logits for the next token, so we squeeze the logits to remove the draft size dimension
                next_token_logits = next_token_logits.squeeze(dim=1)

            sampler_output: SamplerOutput = self.sampler.sample(
                inputs, next_token_logits
            )
            next_tokens = sampler_output.next_tokens

            next_tokens = self._postprocess_speculated_tokens(
                next_tokens, speculated_inputs
            )

            # Update the inputs, but take into account that if the sequence is finished,
            # we should not update it with the next token, but with the padding token.
            next_tokens = next_tokens * unfinished_sequences.reshape(
                -1, 1
            ) + self.sampling_config.pad_token_id * (
                1 - unfinished_sequences.reshape(-1, 1)
            )

            # Update the inputs with the next tokens
            if self.text_streamer:
                self.text_streamer.put(next_tokens)

            inputs = torch.cat([inputs, next_tokens], dim=-1)

            # Update probs and logprobs for return
            probs, logprobs = self._update_probs_logprobs(
                probs, logprobs, sampler_output
            )

            # Check if the sequence is finished, using the stopping criteria
            stopping_criteria_output = self.stopping_criteria.stop_now(inputs)
            unfinished_sequences = unfinished_sequences & ~stopping_criteria_output

            # Update the attention mask (REDO)
            self.attention_mask = self._adjust_mask_for_generation(
                self.attention_mask, unfinished_sequences, next_tokens.shape[-1]
            )

        if self.text_streamer:
            self.text_streamer.end()

        if self.pard_config:
            PACE_LLM_DEBUG(
                f"Mean accepted tokens: {sum(self.total_speculated_tokens) / len(self.total_speculated_tokens)}"
            )
            PACE_LLM_DEBUG(
                f"Accuracy of speculated tokens: {sum(self.total_speculated_tokens) / inputs.shape[1] * 100:.2f}%"
            )
        return self._prepare_output_for_generate(
            output_token_ids=inputs,
            input_logprobs=input_logprobs,
            logprobs=logprobs,
            probs=probs,
        )

    def generate(self, inputs: torch.Tensor) -> GeneratorOutput:
        """
        Generates text from the model given the input prompts.
        NOTE: prepare_for_generate should be called before calling this method.

        Args:
            inputs (torch.Tensor): The input prompts

        Returns:
            GeneratorOutput: The generated outputs
        """

        if self.sampling_config.sampling_mode == SamplingMode.BEAM_SEARCH:
            return self._beam_search(inputs)
        else:
            return self._generate(inputs)

    def __repr__(self):
        return f"Generator(model_path={self.model_path}, tokenizer_path={self.tokenizer_path})"

    def get_tokenizer(
        self,
    ) -> PreTrainedTokenizer:
        """
        Returns the tokenizer

        Returns:
            PreTrainedTokenizer: The tokenizer
        """
        return self.tokenizer

    def get_config(self) -> PretrainedConfig:
        """
        Returns the model config

        Returns:
            PretrainedConfig: The model config
        """
        return self.model.get_config()
