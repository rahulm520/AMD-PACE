# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************
import os
import math
from enum import Enum
from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple, List

import torch
from transformers import PretrainedConfig

from pace.utils.logging import PACE_LLM_ASSERT, PACE_LLM_DEBUG


class KVCacheType(Enum):
    """
    Enum class for the different types of key-value caches that can be used.
    """

    DYNAMIC = "dynamic"
    BMC = "bmc"

    @staticmethod
    def get_kv_cache_type(cache_type: Union[str, "KVCacheType"]) -> "KVCacheType":
        """
        Returns the KVCacheType enum corresponding to the provided cache type.
        """
        if isinstance(cache_type, str):
            return KVCacheType(cache_type)
        return cache_type


def get_kv_cache_class(cache_type: Union[str, KVCacheType]) -> "KVCacheBase":
    """
    Returns the KVCacheBase class corresponding to the provided cache type.
    """
    cache_type = KVCacheType.get_kv_cache_type(cache_type)
    if cache_type == KVCacheType.DYNAMIC:
        return DynamicKVCache
    elif cache_type == KVCacheType.BMC:
        return BMCKVCache
    raise ValueError(f"Invalid cache type: {cache_type}")


class KVCacheManager:
    """
    Manages key-value caches for multiple layers.
    """

    def __init__(
        self, config: PretrainedConfig, max_seq_length: int, cache_type: KVCacheType
    ):
        """
        Initialize the KVCacheManager.

        Args:
            config (PretrainedConfig): Configuration object containing model parameters.
            max_seq_length (int): Maximum sequence length for the cache.
            cache_type (KVCacheType): Type of key-value cache to use.
        """
        self.num_layers = config.num_hidden_layers
        self.cache_type = cache_type
        self.kv_cache_class = get_kv_cache_class(self.cache_type)
        # Initialize cache objects for each layer
        self.cache_objects: List[KVCacheBase] = [
            self.kv_cache_class(max_seq_length) for _ in range(self.num_layers)
        ]

    def __len__(self) -> int:
        """
        Return the current sequence length (shared across layers).

        Returns:
            int: Current sequence length.
        """
        return int(self.cache_objects[0].seq_len)

    def update_mask(self, attn_mask: torch.Tensor, input_size: int) -> torch.Tensor:
        """
        Update the attention mask for the cache.

        Args:
            attn_mask: The attention mask to update.
            inputs: The inputs used to update the mask.
        Returns:
            Updated attention mask.
        """
        result = self.cache_objects[0].update_mask(attn_mask, input_size)
        return result

    def __getitem__(self, idx: int) -> "KVCacheBase":
        """
        Get the cache object for a specific layer.

        Args:
            idx (int): Index of the layer.

        Returns:
            Cache object for the specified layer.
        """
        return self.cache_objects[idx]

    def update_cache_with_beam_indices(self, beam_idx: torch.Tensor) -> None:
        """
        Updates the key-value caches based on the beam indices (only for beam search).
        This is required since beam search will alter the previous tokens and we need to update the kv_caches
        accordingly.

        Args:
            beam_idx (torch.Tensor): The beam indices
        """
        for cache_object in self.cache_objects:
            cache_object.update_cache_with_beam_indices(beam_idx)

    def remove_cache(self, remove_len: int):
        """
        Remove the last `remove_len` tokens from the key-value caches.
        """
        for cache_object in self.cache_objects:
            cache_object.remove_cache(remove_len)


class KVCacheBase(ABC):
    """
    Abstract base class for key-value caches.
    """

    def __init__(self):
        pass

    @abstractmethod
    def update_cache(
        self, key_states: torch.Tensor, value_states: torch.Tensor, concat_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Abstract method for updating the key and value caches.
        Subclasses must provide their own implementation.
        """
        pass

    @abstractmethod
    def remove_cache(self, remove_len: int) -> None:
        """
        Abstract method for removing the last `remove_len` tokens from the key-value caches.
        Subclasses must provide their own implementation.
        """
        pass

    def update_mask(self, attn_mask: torch.Tensor, input_size: int) -> torch.Tensor:
        """
        Update the attention mask after each iteration.

        Could be overridden by subclasses to implement specific mask
        update logic.
        """
        return attn_mask

    def update_cache_with_beam_indices(self, beam_idx: torch.Tensor) -> None:
        """
        Updates the key-value caches based on the beam indices.
        """
        if self.key is not None and self.value is not None:
            self.key = self.key.index_select(0, beam_idx.to(self.key.device))
            self.value = self.value.index_select(0, beam_idx.to(self.value.device))


class BMCKVCache(KVCacheBase):
    """
    Key-value cache implementation for BMC type.
    """

    def __init__(self, max_seq_length: int):
        super().__init__()
        num_splits = int(
            os.getenv("PACE_BMC_NUM_SPLITS", int(math.sqrt(max_seq_length)))
        )
        # Example: 2048 // 32 = 64
        self.tokens_per_split = max_seq_length // num_splits
        self.key = None
        self.value = None
        self.seq_len = 0
        self.concat_dim = (
            None  # Dimension along which the key and value tensors are concatenated
        )

        PACE_LLM_DEBUG(
            f"BMCKVCache initialized with {num_splits} splits, "
            f"tokens per split: {self.tokens_per_split}, "
            f"max sequence length: {max_seq_length}"
        )

    def update_mask(self, attn_mask: torch.Tensor, input_size: int) -> torch.Tensor:
        """
        Update the attention mask by appending padding values.

        Args:
            attn_mask (torch.Tensor): The current attention mask.
            input_size (int): The size of the input sequence.

        Returns:
            torch.Tensor: The updated attention mask with padding.
        """
        # TODO: Can be used to update mask from generator class @first_pass can be a param to identify first pass
        # if not first_pass:
        #     attn_mask = torch.cat(
        #         [
        #             attn_mask[:, : inputs.shape[-1] - 1],
        #             attn_mask.new_ones((attn_mask.shape[0], 1)),
        #         ],
        #         dim=-1,
        #     )
        segment_idx = (input_size - 1) // self.tokens_per_split
        shape = list(attn_mask.shape)
        shape[-1] = (segment_idx + 1) * self.tokens_per_split - input_size
        min_val = torch.finfo(attn_mask.dtype).min
        inf_padding = torch.full(
            shape, min_val, dtype=attn_mask.dtype, device=attn_mask.device
        )
        return torch.cat([attn_mask, inf_padding], dim=-1)

    def remove_cache(self, remove_len: int) -> None:
        """
        Remove the last `remove_len` tokens from the key-value caches.
        This method modifies the key and value tensors in place.

        Args:
            remove_len (int): The number of tokens to remove from the cache.

        Returns:
            None: The function modifies the key and value tensors in place.
        """
        if self.key is not None and self.value is not None:
            # Ensure that we do not remove more tokens than available
            if remove_len > self.seq_len:
                raise ValueError("Cannot remove more tokens than available in cache.")
            # Update the instance sequence length
            self.seq_len -= remove_len

    def _create_new_segment(self, shape: Tuple[int], dtype: torch.dtype):
        """
        Create new segments of key and value tensors by appending and
        potentially copying from existing cached tensors.

        Args:
        - shape (List[int]): The shape of the new segment to be created.
        - dtype (torch.dtype): The data type of the new segment tensors.

        Returns:
            None: The function modifies the key and value tensors in place.
        """
        # Create new tensors initialized to zero with the determined shape.
        new_key = torch.zeros(shape, dtype=dtype)
        new_value = torch.zeros(shape, dtype=dtype)
        # Copy existing keys and values into the new segment if available
        if self.key is not None and self.value is not None:
            new_key.narrow(self.concat_dim, 0, self.key.size(self.concat_dim)).copy_(
                self.key
            )
            new_value.narrow(
                self.concat_dim, 0, self.value.size(self.concat_dim)
            ).copy_(self.value)
        self.key = new_key
        self.value = new_value

    def update_cache(
        self, key_states: torch.Tensor, value_states: torch.Tensor, concat_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the instance-level key and value tensors and manage class-level sequence length.

        Parameters:
        - key_states (torch.Tensor): The tensor containing the key states to be updated.
        - value_states (torch.Tensor): The tensor containing the value states to be updated.
        - concat_dim (int): The dimension along which the tensors are concatenated.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Updated key and value tensors.
        """

        if self.concat_dim is None:
            self.concat_dim = concat_dim
        else:
            PACE_LLM_ASSERT(
                self.concat_dim == concat_dim,
                "concat_dim should be the same for all updates",
            )

        # Update the instance-level sequence length
        updated_seq_len = self.seq_len + key_states.size(self.concat_dim)

        # Check if a new segment needs to be created
        # Determine if a new segment needs to be created
        new_segment_needed = (
            # If key or value is not initialized
            (self.key is None or self.value is None)
            # Or if the current segment does not have enough space
            or (updated_seq_len >= self.key.size(self.concat_dim))
        )

        if new_segment_needed:
            PACE_LLM_DEBUG(
                f"Creating new segment for key and value tensors. "
                f"Updated sequence length: {updated_seq_len}, "
                f"Current allocated sequence length: {self.key.size(self.concat_dim) if self.key is not None else 0}, "
                f"Segment index: {(updated_seq_len - 1) // self.tokens_per_split}"
            )
            # Calculate the segment index based on the current sequence length
            segment_idx = (updated_seq_len - 1) // self.tokens_per_split
            # Create a new segment with the appropriate shape
            new_shape = list(key_states.shape)
            new_shape[self.concat_dim] = (segment_idx + 1) * self.tokens_per_split
            self._create_new_segment(tuple(new_shape), key_states.dtype)

        # Copy the new key and value states into the last position of the key and value tensors
        if self.key is not None and self.value is not None:
            self.key.narrow(
                self.concat_dim, self.seq_len, key_states.size(concat_dim)
            ).copy_(key_states)
            self.value.narrow(
                self.concat_dim, self.seq_len, value_states.size(concat_dim)
            ).copy_(value_states)
        self.seq_len = updated_seq_len  # Update the sequence length
        return self.key, self.value


class DynamicKVCache(KVCacheBase):
    """
    Key-value cache implementation for Dynamic type.
    """

    def __init__(self, max_seq_length):
        """
        Initialize the DynamicKVCache with optional key and value tensors.

        Args:
            config (PretrainedConfig): Configuration for the model.
            max_seq_length (int): Maximum sequence length for the cache.
        """
        super().__init__()
        self.key: Optional[torch.Tensor] = None  # Initialize key tensor as None
        self.value: Optional[torch.Tensor] = None  # Initialize value tensor as None
        self.seq_len = 0  # Initialize the sequence length to 0
        self.concat_dim = None  # Initialize the concatenation dimension as None

        PACE_LLM_DEBUG(
            f"DynamicKVCache initialized with max sequence length: {max_seq_length}"
        )

    def remove_cache(self, remove_len: int) -> None:
        """
        Remove the last `remove_len` tokens from the key-value caches.
        This method modifies the key and value tensors in place.
        Args:
            remove_len (int): The number of tokens to remove from the cache.
        Returns:
            None: The function modifies the key and value tensors in place.
        """
        if self.key is not None and self.value is not None:
            # Ensure that we do not remove more tokens than available
            if remove_len > self.seq_len:
                raise ValueError("Cannot remove more tokens than available in cache.")
            # Update the sequence length
            self.seq_len -= remove_len
            self.key = self.key.narrow(self.concat_dim, 0, self.seq_len)
            self.value = self.value.narrow(self.concat_dim, 0, self.seq_len)

    def update_cache(
        self, key_states: torch.Tensor, value_states: torch.Tensor, concat_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches with new states.

        Args:
            key_states (torch.Tensor): New key states to be added to the cache.
            value_states (torch.Tensor): New value states to be added to the cache.
            concat_dim (int): Dimension along which to concatenate the new states.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated key and value states.
        """
        if self.concat_dim is None:
            self.concat_dim = concat_dim
        else:
            PACE_LLM_ASSERT(
                self.concat_dim == concat_dim,
                "concat_dim should be the same for all updates",
            )
        if self.key is not None and self.value is not None:
            # Concatenate old and new key states
            self.key = torch.cat([self.key, key_states], dim=concat_dim)
            # Concatenate old and new value states
            self.value = torch.cat([self.value, value_states], dim=concat_dim)
        else:
            self.key = key_states  # Update the key cache
            self.value = value_states  # Update the value cache
        self.seq_len = int(self.key.size(concat_dim))  # Update the sequence length
        return self.key, self.value
