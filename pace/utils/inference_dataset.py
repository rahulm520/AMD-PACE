# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from transformers import (
    PreTrainedTokenizer,
)
from datasets import load_dataset, Dataset
import json
from typing import Optional
import random
import numpy as np
import torch


class LLMInferenceDataset:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_name: Optional[str] = None,
        split: str = "test",
        custom_data_path: Optional[str] = None,
        version: str = None,
        max_input_len: int = 1024,
        max_target_len: int = 1024,
        input_field: str = "data",
        target_field: Optional[str] = None,
        prompt_type: Optional[str] = None,
        num_of_samples: Optional[int] = None,
        seed: Optional[int] = 123,
    ):
        """
        Initializes and tokenizes either a Hugging Face dataset, custom list of dicts, or JSON file.

        :param tokenizer: HuggingFace tokenizer instance.
        :param dataset_name: Name of the HuggingFace dataset (e.g., 'cnn_dailymail', 'gsm8k').
        :param custom_data_path: Path to a JSON/JSONL file with list of dicts.
        :param split: Dataset split to load (ignored for custom data).
        :param version: Dataset configuration/version (used for HuggingFace datasets).
        :param max_input_len: Max token length for input.
        :param max_target_len: Max token length for target.
        :param input_field: Field name to use as input text.
        :param target_field: Field name to use as target/label text.
        :param prompt_type: Field Name to be used for chat template.
        """
        self.tokenizer = tokenizer
        self.input_field = input_field
        self.target_field = target_field
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Load dataset
        if dataset_name:
            if version:
                raw_dataset = load_dataset(dataset_name, version, split=split)
            else:
                raw_dataset = load_dataset(dataset_name, split=split)

        elif custom_data_path:
            with open(custom_data_path, "r", encoding="utf-8") as f:
                if custom_data_path.endswith(".json"):
                    data = json.load(f)  # a list of dicts
                elif custom_data_path.endswith(".jsonl"):
                    data = [json.loads(line) for line in f]
                else:
                    raise ValueError("Unsupported file format. Use .json or .jsonl")
            raw_dataset = Dataset.from_list(data)

        else:
            raise ValueError(
                "You must specify either 'dataset_name', 'custom_data', or 'custom_data_path'."
            )

        # Pre-tokenize all samples and store them
        self.samples = []
        for item in raw_dataset:
            input_text = item[input_field]
            if prompt_type == "llama3":
                system = "You are a helpful assistant"
                input_text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            elif prompt_type == "r1":
                messages = [{"role": "user", "content": input_text}]
                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            elif prompt_type == "qwen":
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input_text},
                ]
                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            input_enc = tokenizer(
                input_text,
                add_special_tokens=False,
                # padding=False,
                return_tensors=None,
            )
            if target_field and target_field in item:
                target_text = item[target_field]
                target_enc = tokenizer(
                    target_text,
                    max_length=max_target_len,
                    truncation=True,
                    # padding=False,
                    return_tensors=None,
                )
                labels = target_enc["input_ids"]
            else:
                target_text = None
                labels = []

            self.samples.append(
                {
                    "input_ids": input_enc["input_ids"],
                    "attention_mask": input_enc["attention_mask"],
                    "labels": labels,
                    "input_text": input_text,
                    "target_text": target_text,
                }
            )

        if num_of_samples:
            if num_of_samples > len(self.samples):
                raise IndexError(
                    "Number of samples requested are greater than the dataset size."
                )

            self.total = num_of_samples
        else:
            self.total = len(self.samples)
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.total:
            raise StopIteration
        item = self.samples[self.index]
        self.index += 1
        return item

    def get_item(self, index):
        if index < 0 or index >= self.total:
            raise IndexError("Index out of range")
        return self.samples[index]

    def get_batch(self, batch_size: int, start_index: int = 0):
        """
        Return a batch of tokenized samples with dynamic padding.

        :param batch_size: Number of samples to include in the batch.
        :param start_index: Starting index of the batch.
        :return: A dictionary of batched tensors with padding.
        """
        end_index = min(start_index + batch_size, self.total)
        if start_index >= self.total:
            raise IndexError("Start index exceeds dataset size.")

        batch = self.samples[start_index:end_index]

        # Extract unbatched examples
        input_batch = [
            {"input_ids": s["input_ids"], "attention_mask": s["attention_mask"]}
            for s in batch
        ]
        label_batch = [{"input_ids": s["labels"]} for s in batch]

        # Pad inputs and labels dynamically
        input_padded = self.tokenizer.pad(
            input_batch, padding=True, return_tensors="pt"
        )
        label_padded = self.tokenizer.pad(
            label_batch, padding=True, return_tensors="pt"
        )

        return {
            "input_ids": input_padded["input_ids"],
            "attention_mask": input_padded["attention_mask"],
            "labels": label_padded["input_ids"],
            "input_texts": [s["input_text"] for s in batch],
            "target_texts": [s["target_text"] for s in batch],
        }

    def size(self):
        return self.total

    def reset(self):
        self.index = 0
