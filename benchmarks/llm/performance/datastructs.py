# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from pace.utils.logging import PACE_LLM_ASSERT


@dataclass
class ModelArgs:
    model_name: str
    dtype: str
    llm_operators: Optional[dict] = None
    spec_config: Optional[dict] = None


@dataclass
class GenerationArgs:
    input_tokens: int
    output_tokens: int
    batch_size: int
    num_beams: int
    kv_cache_type: str
    do_sample: bool
    manual_seed: int


@dataclass
class TokenArgs:
    time_to_first_token: bool = False
    time_per_tokens: bool = False


@dataclass
class BenchmarkArgs:
    frameworks: List[str]
    model_args: ModelArgs
    generation_args: GenerationArgs
    use_real_data: bool
    num_runs: int
    warmup_runs: int
    visualize: bool
    verbose: bool
    output_dir: str
    token_args: TokenArgs
    system_metrics: bool


class BaseData(ABC):

    @abstractmethod
    def to_dict(self):
        pass

    def __str__(self):
        # remove None values
        dict_val = self.to_dict()
        dict_val = {k: v for k, v in dict_val.items() if v is not None}

        return json.dumps(dict_val, indent=4)

    def __repr__(self):
        return str(self)

    def dump(self, filename):
        json.dump(self.to_dict(), open(filename, "w"), indent=4)


@dataclass
class GeneraterOutput:
    total_time: float
    input_tokens: int
    total_tokens: int
    ttft: Optional[float]
    time_per_tokens: Optional[List[float]] = None
    mean_accepted_tokens: Optional[float] = None


class GeneratorOutputAggregator:

    def __init__(self, token_args: TokenArgs = TokenArgs()):
        self.total_generation_times = []
        self.generated_tokens_count = 0
        self.total_tokens_count = 0

        # If token_metrics is True, we will store the time taken to generate each token
        self.token_args = token_args
        self.ttft_times = []
        self.time_per_tokens = []
        self.mean_accepted_tokens = []

    def append(self, generation_output: GeneraterOutput):
        self.total_generation_times.append(generation_output.total_time)
        # Count newly generated tokens
        self.generated_tokens_count += (
            generation_output.total_tokens - generation_output.input_tokens
        )
        # Count total tokens generated
        self.total_tokens_count += generation_output.total_tokens

        if self.token_args.time_to_first_token:
            self.ttft_times.append(generation_output.ttft)
        if self.token_args.time_per_tokens:
            self.time_per_tokens.append(generation_output.time_per_tokens)
        if generation_output.mean_accepted_tokens is not None:
            self.mean_accepted_tokens.append(generation_output.mean_accepted_tokens)


@dataclass
class Metrics(BaseData):
    total_tokens: int
    total_gen_tokens: int
    average_gen_time: float
    average_latency_per_token: float
    total_tps: float
    output_tps: float
    average_ttft: Optional[float] = None
    time_per_tokens: Optional[List[float]] = None
    mean_accepted_tokens: Optional[float] = None

    def to_dict(self):
        return {
            "total_tokens": self.total_tokens,
            "total_gen_tokens": self.total_gen_tokens,
            "average_gen_time": self.average_gen_time,
            "average_latency_per_token": self.average_latency_per_token,
            "total_tps": self.total_tps,
            "output_tps": self.output_tps,
            "average_ttft": self.average_ttft,
            "time_per_tokens": self.time_per_tokens,
            "mean_accepted_tokens": self.mean_accepted_tokens,
        }


@dataclass
class SystemMetrics(BaseData):
    interval: float
    cpu_usage: List[float]
    ram_usage: List[float]
    peak_ram_usage: float

    def to_dict(self):
        return {
            "interval": self.interval,
            "cpu_usage": self.cpu_usage,
            "ram_usage": self.ram_usage,
            "peak_ram_usage": self.peak_ram_usage,
        }


@dataclass
class BenchmarkResults(BaseData):
    framework: str
    model_args: ModelArgs
    generation_args: GenerationArgs
    num_runs: int
    warmup_runs: int
    metrics: Metrics
    system_metrics: Optional[SystemMetrics] = None

    def to_dict(self):
        return {
            "framework": self.framework,
            "model_name": self.model_args.model_name,
            "dtype": str(self.model_args.dtype),
            "llm_operators": self.model_args.llm_operators,
            "generation_args": {
                "input_tokens": self.generation_args.input_tokens,
                "output_tokens": self.generation_args.output_tokens,
                "batch_size": self.generation_args.batch_size,
                "num_beams": self.generation_args.num_beams,
                "do_sample": self.generation_args.do_sample,
                "manual_seed": self.generation_args.manual_seed,
            },
            "num_runs": self.num_runs,
            "warmup_runs": self.warmup_runs,
            "metrics": self.metrics.to_dict(),
            "system_metrics": (
                self.system_metrics.to_dict() if self.system_metrics else None
            ),
        }


@dataclass
class BenchmarkResultsList(BaseData):
    """A class to aggregate and hold a list of BenchmarkResults."""

    results: List[BenchmarkResults] = field(default_factory=list)

    def append(self, benchmark_results: BenchmarkResults):
        if self.results:
            first_result = self.results[0]
            PACE_LLM_ASSERT(
                first_result.model_args == benchmark_results.model_args,
                "Cannot append BenchmarkResults with different model_args",
            )
            PACE_LLM_ASSERT(
                first_result.generation_args == benchmark_results.generation_args,
                "Cannot append BenchmarkResults with different generation_args",
            )
            PACE_LLM_ASSERT(
                first_result.num_runs == benchmark_results.num_runs,
                "Cannot append BenchmarkResults with different num_runs",
            )
            PACE_LLM_ASSERT(
                first_result.warmup_runs == benchmark_results.warmup_runs,
                "Cannot append BenchmarkResults with different warmup_runs",
            )
        self.results.append(benchmark_results)

    def to_dict(self):
        if not self.results:
            return {}

        first_result = self.results[0]
        return_dict = {
            "model_name": first_result.model_args.model_name,
            "dtype": str(first_result.model_args.dtype),
            "llm_operators": first_result.model_args.llm_operators,
            "num_runs": first_result.num_runs,
            "warmup_runs": first_result.warmup_runs,
            "generation_args": {
                "input_tokens": first_result.generation_args.input_tokens,
                "output_tokens": first_result.generation_args.output_tokens,
                "batch_size": first_result.generation_args.batch_size,
                "num_beams": first_result.generation_args.num_beams,
                "do_sample": first_result.generation_args.do_sample,
                "manual_seed": first_result.generation_args.manual_seed,
            },
            "benchmark_results": [
                {
                    "framework": res.framework,
                    "metrics": res.metrics.to_dict(),
                    "system_metrics": (
                        res.system_metrics.to_dict() if res.system_metrics else None
                    ),
                }
                for res in self.results
            ],
        }
        return return_dict
