# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************
# In /ZenDNN_PACE/tests run using python -m unittest -v llm_infra/test_cache.py

from torch.testing._internal.common_utils import TestCase
import torch
from transformers import PretrainedConfig
from pace.llm.cache import KVCacheManager, BMCKVCache, DynamicKVCache, KVCacheType


class MockConfig(PretrainedConfig):
    """Mock configuration for testing purposes."""

    def __init__(self, num_hidden_layers=2):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers


class TestKVCacheManager(TestCase):
    def setUp(self):
        self.config = MockConfig(num_hidden_layers=2)
        self.max_seq_length = 128
        self.kv_cache_manager = KVCacheManager(
            self.config, self.max_seq_length, KVCacheType.DYNAMIC
        )

    def test_initialization(self):
        self.assertEqual(len(self.kv_cache_manager.cache_objects), 2)
        self.assertIsInstance(self.kv_cache_manager.cache_objects[0], DynamicKVCache)

    def test_getitem(self):
        cache_layer = self.kv_cache_manager[0]
        self.assertIsInstance(cache_layer, DynamicKVCache)

    def test_update_mask(self):
        attn_mask = torch.ones((2, 10))
        inputs = torch.ones((2, 10))
        updated_mask = self.kv_cache_manager.update_mask(attn_mask, inputs.size(1))
        self.assertEqual(updated_mask.shape, (2, 10))  # Should append an extra column


class TestBMCKVCache(TestCase):
    def setUp(self):
        self.config = MockConfig(num_hidden_layers=2)
        self.max_seq_length = 128
        self.bmc_cache = BMCKVCache(self.max_seq_length)
        self.key_states = torch.rand(2, 10, 64)
        self.value_states = torch.rand(2, 10, 64)

    def test_update_cache(self):
        key, value = self.bmc_cache.update_cache(
            self.key_states, self.value_states, concat_dim=1
        )
        self.assertEqual(key.shape[1], 11)  # Should match the input sequence length
        self.assertEqual(value.shape[1], 11)

    def test_update_mask(self):
        attn_mask = torch.ones((2, 10))
        inputs = torch.ones((2, 10))
        updated_mask = self.bmc_cache.update_mask(attn_mask, inputs.size(1))
        self.assertEqual(updated_mask.shape, (2, 11))  # Should add padding


class TestDynamicKVCache(TestCase):
    def setUp(self):
        self.config = MockConfig(num_hidden_layers=2)
        self.max_seq_length = 128
        self.Dynamic_cache = DynamicKVCache(self.max_seq_length)
        self.key_states = torch.rand(2, 10, 64)
        self.value_states = torch.rand(2, 10, 64)

    def test_update_cache(self):
        key, value = self.Dynamic_cache.update_cache(
            self.key_states, self.value_states, concat_dim=1
        )
        self.assertEqual(key.shape[1], 10)  # Should match the input sequence length
        self.assertEqual(value.shape[1], 10)
