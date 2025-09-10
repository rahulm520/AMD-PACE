# AMD PACE - AMD Platform Aware Compute Engine

To meet the demands of rapidly advancing research, we’re introducing AMD PACE — an inference-serving solution for high-performance LLMs on AMD platforms. AMD PACE makes it fast and easy to integrate research ideas and accelerate real-time deployment.

Engineered for AMD CPUs with AVX512 support, AMD PACE is a PyTorch extension that provides a framework for developing and testing novel kernel implementations and graph-level optimizations.

> NOTE: AMD PACE is designed and tested for systems with AVX512 or higher support. On systems lacking AVX512, performance may degrade significantly due to fallback to slower reference implementations, or the library might not function as intended.

## About
 * CPU-Optimized Inference: Engineered for CPU LLM workloads, AMD PACE delivers measurable performance gains over existing inference serving solutions such as vLLM using CPU friendly cache and kernel optimizations.

 * Speculative Decoding: AMD PACE features a built-in implementation of PARallel Draft Model Adaptation (PARD), a speculative decoding technique that can deliver up to 5× throughput improvement versus a standard autoregressive baseline. More in [SpeculativeDecoding.md](docs/SpeculativeDecoding.md). You can easily enable PARD by providing a `PardSpecDecodeConfig` when initializing the `LLMModel`, as shown in the [example](examples/pace_llm_pard.py) .

 * Data Parallelism: Integrated data-parallel support significantly accelerates both speculative decoding and standard inference, enabling end-to-end speedups of up to 25×. AMD PACE achieves this by serving multiple requests concurrently across multiple model instances. The examples [Multi-Instance](examples/multi_instance_pace.py) and [Speculative Decoding Multi-Instance](examples/multi_instance_sd_pace.py) demonstrates this architecture, where a pool of worker processes handles incoming requests in parallel, maximizing hardware utilization and overall throughput.

 * Ongoing Improvements: AMD PACE is designed to evolve with research needs and emerging production workloads. Its core mission is to serve as a research vehicle, providing a flexible and extensible framework for exploring forward-looking hardware optimizations and integrating new ideas from the fast-moving field of AI.

## Contents

* [Installation](#installation)
* [Inference Server](docs/InferenceServer.md)
* [More about AMD PACE](docs/Plugin.md)
* [Models Supported](#models-supported)
* [Performance Guide](docs/PerformanceGuide.md)
* [Benchmarks](#benchmarks)
* [Contributing to AMD PACE](docs/Contributing.md)
* [Tests](tests/README.md)
* [External Dependencies](#external-dependencies)

## Installation
To install AMD PACE, follow the instructions below:

> NOTE: AMD PACE will need gcc>=12, make and ccache (for ZenDNN build) installed.
>
> On ubuntu, they can be installed with `sudo apt install build-essential gcc-12 g++-12 ccache`

1. We recommend to use miniforge environment for installing AMD PACE. Install miniforge from [here](https://conda-forge.org/miniforge/).
Once miniforge is installed, create a environment with python 3.12 as follows:
    ```
    conda create -n pace-env-py3.12 python=3.12 -y
    conda activate pace-env-py3.12
    ```

    NOTE: AMD PACE is tested to work with Python 3.9 and above. Python 3.12 is recommended for the best compatibility with dependencies.

1. Install the required dependencies for AMD PACE as follows:
    ```
    pip install -r requirements.txt
    ```

1. Build AMD PACE from source as follows:
    ```
    pip install -r build_requirements.txt [-v] .
    ```
    This will build AMD PACE and install it in the current environment. The `-v` option is optional and can be used to enable verbose output during the build process.

    > NOTE: It uses the new way of building packages with `pip`, for more details refer to [PEP 517](https://peps.python.org/pep-0517/). The `build_requirements.txt` should be passed in during installation to ensure that the build environment is set up correctly, please refer to [PEP 518](https://peps.python.org/pep-0518/) for more details.

    For developers who need to build AMD PACE frequently, using pip with `--no-build-isolation` is recommended to avoid unnecessary overhead of creating isolated environments for each build. This speeds up the build process significantly. Make sure to have all the required dependencies installed in your environment before using this option.
    ```
    pip install --no-build-isolation [-v] .
    ```

    > NOTE: Building AMD PACE, especially the oneDNN component, can require significant memory. If your system does not have enough RAM, the build process may fail or your machine may run out of memory.

## Models Supported
The following models are supported by AMD PACE:
1. [INT8 DLRMv2](docs/DLRMv2.md)
1. [Large Language Models](docs/LLM.md)
    1. [Speculative Decoding](docs/SpeculativeDecoding.md)
    1. [LLM Benchmarks](benchmarks/llm/performance/README.md)
    1. [LLM Evaluation](benchmarks/llm/accuracy/README.md)

## Benchmarks
Benchmarks for AMD PACE are available in the `benchmarks` directory. The benchmarks include:
* [LLM Performance](benchmarks/llm/performance/README.md)
* [LLM Accuracy](benchmarks/llm/accuracy/README.md)

## Verbose
To enable verbose mode, set the environment variable `PACE_LOG_LEVEL`. The following levels are supported:

| Level   |  Environment Variable           |
|---------|---------------------------------|
| Debug   | `export PACE_LOG_LEVEL=debug`   |
| Profile | `export PACE_LOG_LEVEL=profile` |
| Info    | `export PACE_LOG_LEVEL=info`    |
| Warning | `export PACE_LOG_LEVEL=warning` |
| Error   | `export PACE_LOG_LEVEL=error`   |

NOTE: By default, the log level is set to `info`.

## External Dependencies
AMD PACE depends on the following libraries:

| Library        | Version  |
|----------------|----------|
| PyTorch        | v2.7.0   |
| OneDNN         | v3.8     |
| FBGEMM         | v1.2.0   |
