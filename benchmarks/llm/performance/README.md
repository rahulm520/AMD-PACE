# Benchmarking Large Language Models

## Contents
1. [Scenarios](#scenarios)
    1. [Offline](#offline)
2. [Benchmarking Infrastructure](#benchmarking-infrastructure)
    1. [Data Generator](#data-generator)
    2. [Visualizations](#visualizations)
    3. [System Metrics](#system-metrics)
    4. [Token Metrics](#token-metrics)

## Scenarios

### Offline

This document provides an overview of the benchmarking process for language models using the `benchmark_llm_offline.py` script. The script benchmarks the offline performance (throughput and latency) of language models using different frameworks, with different configurations and data types.

#### Configuration

The benchmarking process is configured using a JSON file. Below is an example configuration file:

```json
{
    "frameworks": [
        "pace",
        "hf"
    ],
    "model_args": {
        "model_name": "meta-llama/Llama-3.1-8B",
        "dtype": "bf16",
        "llm_operators": {
            "Norm": "NATIVE",
            "QKVProjection": "TPP",
            "Attention" : "JIT",
            "OutProjection": "TPP",
            "MLP": "TPP",
            "LMHead": "TPP"
        },
        "spec_config": {
            "model_name": "amd/PARD-Llama-3.2-1B",
            "num_speculated_tokens": 12
        }
    },
    "use_real_data": true,
    "generation_args": {
        "input_tokens": 128,
        "output_tokens": 128,
        "batch_size": 1,
        "num_beams": 1,
        "kv_cache_type": "BMC",
        "do_sample": false,
        "manual_seed": 0
    },
    "warmup_runs": 2,
    "num_runs": 5,
    "visualize": false,
    "verbose": true,
    "output_dir": "./benchmark_results",
    "token_metrics": {
        "time_to_first_token": false,
        "time_per_tokens": false
    },
    "system_metrics": false
}
```

#### Configuration Parameters

- `frameworks`: List of frameworks to benchmark (e.g., `"pace"`, `"hf"`).
- `model_args`: Dictionary of model arguments, including:
    - `model_name`: Name of the model to benchmark (e.g., `"facebook/opt-125m"`).
    - `dtype`: Data type to use (e.g., `"bf16"`).
    - `llm_operators`: Dictionary specifying operator implementations (e.g., `"Norm": "NATIVE"`).
- `use_real_data`: Boolean indicating whether to use real data from a dataset.
- `generation_args`: Dictionary of generation parameters, including:
    - `input_tokens`: Number of input tokens.
    - `output_tokens`: Number of output tokens.
    - `batch_size`: Batch size for benchmarking.
    - `num_beams`: Number of beams for beam search.
    - `kv_cache_type`: Type of key-value cache (e.g., `"BMC"`).
    - `do_sample`: Boolean indicating whether to use sampling.
    - `manual_seed`: Random seed for reproducibility.
- `warmup_runs`: Number of warm-up runs before benchmarking.
- `num_runs`: Number of benchmark runs.
- `visualize`: Boolean indicating whether to generate visualizations.
- `verbose`: Boolean indicating whether to print detailed logs.
- `output_dir`: Directory to save benchmark results.
- `token_metrics`: Dictionary specifying token-level metrics to collect (e.g., `time_to_first_token`, `time_per_tokens`).
- `system_metrics`: Boolean indicating whether to collect system-level metrics.

NOTE:
1. Either `verbose` or `output_dir` should be set, otherwise the script will not run.
2. If `visualize` is set to `true`, the script will generate visualizations for the benchmark results, but it will depend on metrics collected, comparison charts will only be generated if multiple frameworks are benchmarked.

#### Usage
To run the benchmarking script, use the following command:

```bash
python benchmark_llm_offline.py --config /path/to/config.json
```

## Benchmarking Infrastructure

### Data Generator

The `data.py` file provides utilities for generating input data for benchmarking. It includes a class `BenchMarkDataGenerator` that can generate random input data or use real data from a dataset.

The `BenchMarkDataGenerator` class is used to generate input data for benchmarking the language model. It can generate random input data or use real data from a specified dataset.

If using real data, it loads and tokenizes the dataset using the provided tokenizer, and is saved in memory according to the batch size and input tokens. If using random data, it picks random tokens based on the input tokens and batch size from the tokenizer and the same data will be reused across runs. Once the data is generated, it can be accessed using the `next` method, and it will infinitely loop through the data, giving a random batch of input data each time.

NOTE: The data generator cannot be used for evaluation purposes, as it does not guarantee the same data across runs and is not representative of the dataset.

#### Example Usage

```python
from transformers import AutoTokenizer
from data import BenchMarkDataGenerator

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

# Create data generator
data_gen = BenchMarkDataGenerator(
    tokenizer=tokenizer,
    input_tokens=128,
    batch_size=8,
    use_real_data=True
)

# Get data for benchmarking
for _ in range(N):
    data = next(data_gen)
    ...
```

### Visualizations

#### Bar Graphs

The `visualization.py` file provides functions to generate visualizations for the benchmark results. These visualizations help in comparing the performance of different frameworks.

The `create_comparison_bar_graph` function generates bar charts to compare frameworks based on various metrics. The bar graphs are generated for each metric and saved as separate images, comparing the performance of different frameworks. It can be usesful to compare the performance of different frameworks.

The bar graphs compare the following metrics (can be extented):
- `average_gen_time`: Average generation time (seconds)
- `average_latency_per_token`: Average latency (seconds)
- `total_tps`: Total tokens per second
- `output_tps`: Output tokens per second
- `average_ttft`: Average time to first token (seconds)

#### Line Graphs

The `create_comparison_line_graph` function generates line charts to compare frameworks based on token-level and system-level metrics. The line graphs are generated for each metric and saved as separate images, comparing the performance of different frameworks. It can be usesful to compare the performance of different frameworks as well as for one framework over time.

The line graphs compare the following metrics (can be extented):
- `time_per_tokens`: Per-token time (seconds)
- `cpu_usage`: CPU usage (%)
- `ram_usage`: RAM usage (MB)

They can be extended by extending the dictionary in the `create_comparison_line_graph` and `create_comparison_bar_graph` functions.

### System Metrics

The SystemMonitor class is designed to capture real-time CPU and RAM usage in a background thread during benchmarking. This can be useful for understanding the system resource consumption of your benchmarks. It includes methods to start and stop monitoring and provides a usage history over time. The data is collected using psutil library, and ran in the background thread to avoid blocking the main thread.

#### Example Usage

```python
from metrics import SystemMonitor

monitor = SystemMonitor(interval=1)
monitor.start()

# ... run the benchmark ...

monitor.stop()
metrics = monitor.get_history()
print("CPU usage:", metrics.cpu_usage)
print("RAM usage (MB):", metrics.ram_usage)
print("Peak RAM usage (MB):", metrics.peak_ram_usage)
```

NOTE:
1. The system monitor should be off when benchmarking for absolute performance, as it can affect the performance from the overhead of collecting metrics.
2. RAM usage is calculated by the difference between the current memory usage and the memory usage at the start of monitoring. If background processes are running, the RAM usage may not be accurate.


### Token Metrics

1. FirstTokenTimerStreamer: The FirstTokenTimerStreamer class is designed to calculate the time taken for the first time token to be generated during the token generation process.

    #### Example Usage

    ```python
    from metrics import FirstTokenTimerStreamer

    first_token_streamer = FirstTokenTimerStreamer()

    start_time = time.time()
    out = model.generate(inputs, streamer=first_token_streamer, **gen_kwargs)

    ttft = streamer.end_time_first_token - start_time
    ```

2. TokenLatencyStreamer: The TokenLatencyStreamer class is designed to calculate the time taken for each token to be generated during the token generation process.

    ##### Example Usage

    ```python
    from metrics import TokenLatencyStreamer

    token_latency_streamer = TokenLatencyStreamer()

    start_time = time.time()
    out = model.generate(inputs, streamer=token_latency_streamer, **gen_kwargs)

    time_per_tokens = streamer.time_per_tokens()
    ```
