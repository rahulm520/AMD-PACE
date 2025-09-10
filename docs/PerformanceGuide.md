# Performance Guide for AMD PACE

There are multiple factors that can affect the performance of models AMD PACE. This guide provides tips and best practices to help you optimize your setup for the best performance.

## Standard Practices

The standard practices for PyTorch have been mentioned here: [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html). These practices are applicable to AMD PACE as well. Some of them are reiterated below:

#### Utilize Non-Uniform Memory Access (NUMA) Controls
On multi-socket machines, NUMA (Non-Uniform Memory Access) controls can improve performance by optimizing memory locality. For deep learning workloads, binding a process to a specific NUMA node avoids cross-socket memory access, which can reduce overhead and increase throughput.

The following command runs a script on the Nth node, binding both CPU and memory to it:

```bash
numactl --cpunodebind=N --membind=N python <pytorch_script>
```

#### Utilize OpenMP
OpenMP is utilized to bring better performance for parallel computation tasks.

`OMP_NUM_THREADS` is the easiest switch to use for accelerating computations. It determines the number of threads used for OpenMP computations. With the following command, PyTorch will run the task on N OpenMP threads.

```bash
export OMP_NUM_THREADS=N
```

To bind the threads to specific CPU cores, on Linux machines, you can use `GOMP_CPU_AFFINITY`:

```bash
export GOMP_CPU_AFFINITY="0-3"  # Bind to cores 0 to 3
```

#### Optimal Memory Allocator
For deep learning workloads, `jemalloc` or `tcmalloc` can provide better performance than the default `malloc` function by reusing memory more efficiently. AMD PACE recommends using `tcmalloc` for better performance.

To install `tcmalloc` in your current conda environment, run:

```bash
conda install gpertools
```

To use `tcmalloc`, set the environment variable `LD_PRELOAD` to point to the `libtcmalloc.so` library:

```bash
export LD_PRELOAD="${CONDA_PREFIX}/lib/libtcmalloc.so:$LD_PRELOAD"
```

#### THP (Transparent Huge Pages)
Transparent Huge Pages (THP) can improve performance by reducing the overhead of page table management. For models supported by AMD PACE, it is recommended to set THP to `always` for better performance.

To check the current THP setting, run:

```bash
cat /sys/kernel/mm/transparent_hugepage/enabled
```

To set THP to `always`, run (make sure you have the necessary permissions):

```bash
echo always > /sys/kernel/mm/transparent_hugepage/enabled
```

For more information on THP, refer to the [Documentation](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/7/html/performance_tuning_guide/sect-red_hat_enterprise_linux-performance_tuning_guide-configuring_transparent_huge_pages).


## AMD PACE Specific Performance Optimizations

### BMC KV Cache
For Large Language Models, using an optimized key-value cache can significantly improve performance. AMD PACE provides a BMC (Balancing Memory and Compute) KV cache that is optimized for AMD hardware. The core idea is to balance memory usage and compute efficiency, by pre-allocating KV Cache in parts and controlling the memory allocation strategy.

For example, in the dynamic KV cache, the cache is allocated as an when required, and other frameworks pre-allocate the entire cache at the start of the inference. BMC will pre-allocate to a certain size, and then dynamically allocate more memory as needed. This allows for better memory management and can lead to improved performance.

To make use of BMC KV cache, while creating the `LLMModel` object, set `kv_cache_type` to `BMC`:

```python
from pace.llm import LLMModel, KVCacheType

model = LLMModel(
    model_name="your_model_name",
    kv_cache_type=KVCacheType.BMC,
    # other parameters
)
```

By default BMC will decide what is the best size for KV Cache and how many times to pre-allocate. If you want to control it manually, it can be done by setting the environment variable `PACE_BMC_NUM_SPLITS`.

```bash
export PACE_BMC_NUM_SPLITS=4  # Set the number of splits for BMC KV Cache
```
The value of `PACE_BMC_NUM_SPLITS` determines how many times the cache will be allocated. For example, setting `PACE_BMC_NUM_SPLITS=4` means the cache will be allocated in 4 parts, each time doubling the size of the previous allocation until it reaches the maximum size or `PACE_BMC_NUM_SPLITS=1` means the cache will be allocated in a single part.


### AMD PACE Operators
AMD PACE provides a set of optimized operators that can significantly improve performance for specific tasks. These operators are designed to take advantage of AMD hardware capabilities, such as AVX512 instructions and efficient memory access patterns.

PACE employs a modular operator framework that separates an operator's interface (Frontend) from its computational logic (Backend). This design allows for multiple, interchangeable backend implementations for a single operator. For example, a `Linear` operation can be executed by a `NATIVE` backend (using standard PyTorch), a `JIT` backend (using a just-in-time compiled kernel from oneDNN), or a `TPP` backend (using Tensor Processing Primitives). For a detailed explanation of the operator framework, see the [Python Operators documentation](./PythonOps.md).

#### How to Use AMD PACE Operators
To leverage PACE operators with fine-grained control, they can be instantiated directly along with a preferred backend during instantiation of LLMModel. By default, all the operators will use the `NATIVE` backend, which is the standard PyTorch implementation.

For LLM models, there are five main operator types that can be configured:
* `Norm`: Normalization operator
* `QKVProjection`: Query-Key-Value projection operator
* `Attention`: Attention mechanism operator
* `OutProjection`: Output projection operator
* `MLP`: Multi-Layer Perceptron operator
* `LMHead`: Language Model Head operator

To configure these operators, you can use the `OperatorConfig` class. Here is an example of how to set up the operators with specific backends:

```python
from pace.llm import LLMModel, OperatorConfig, LLMOperatorType, LLMBackendType

opconfig = OperatorConfig(
    **{
        LLMOperatorType.Norm: LLMBackendType.NATIVE,
        LLMOperatorType.QKVProjection: LLMBackendType.TPP,
        LLMOperatorType.Attention: LLMBackendType.JIT,
        LLMOperatorType.OutProjection: LLMBackendType.TPP,
        LLMOperatorType.MLP: LLMBackendType.TPP,
        LLMOperatorType.LMHead: LLMBackendType.NATIVE,
    }
)
model = LLMModel(
    model_name="your_model_name",
    operator_config=opconfig,
    # other parameters
)
```
For a more detailed example, please see the [PACE LLM Basic Example](../examples/pace_llm_basic.py).

#### More about backends

1. **NATIVE**: This is the standard PyTorch implementation of the operator. It is the default backend and provides a baseline performance.

    It is suitable for most use cases, but may not be optimized for specific hardware or workloads. If you are not sure which backend to use, starting with the NATIVE backend is a good choice.

1. **JIT**: This backend uses a just-in-time compiled kernel from oneDNN, which can provide performance improvements for certain operations by optimizing the execution path at runtime.

    To use JIT, you can set the backend for the operator in the `OperatorConfig` as shown above. PACE contains a highly optimized JIT backend for the `Attention` operator, which can significantly improve performance for large models. The JIT backend is particularly effective for operations that benefit from dynamic shapes and variable input sizes.

1. **TPP**: This backend uses Tensor Processing Primitives, which make use of libXSMM to provide highly optimized implementations for specific operations. The advantage of TPP comes from blocking the weight matrix which are done at the time of loading the model.

    To use TPP, you can set the backend for the operator in the `OperatorConfig` as shown above. By default, TPP uses uses a blocking size of 32, which is suitable for most models and most scenarios. However, for larger model and prompt heavy workloads, you can experiment with different blocking sizes to find the optimal performance for your specific use case. To set a custom blocking size, you can use the `LIBXSMM_BLOCK_SIZE` environment variable:
    ```bash
    export LIBXSMM_BLOCK_SIZE=64  # Set the blocking size to 64
    ```

1. **IMBPS**: This backend is used for iterative MLP blocks with parameter splits. It is designed to optimize the performance of MLP layers by splitting the parameters across multiple iterations, which can lead to better memory utilization and better TTFT for LLMs. IMBPS is particularly useful for large models where the MLP layers can become a bottleneck. It is not enabled by default, but can be configured in the `OperatorConfig` as follows:
    ```python
    from pace.llm import LLMBackendType, LLMOperatorType, OperatorConfig

    opconfig = OperatorConfig(
        **{
            LLMOperatorType.MLP: LLMBackendType.IMBPS,
        }
    )
    ```

    IMBPS has a parameter that controls the number of splits for the MLP parameters. This can be set using the `IMBPS_BLOCK_SIZE` environment variable:
    ```bash
    export IMBPS_BLOCK_SIZE=4  # Set the number of splits for IMBPS MLP parameters
    ```


## Multi-instance Inference
AMD PACE supports multi-instance inference, which allows you to run multiple instances of a model concurrently, which can significantly improve throughput and resource utilization, even across sockets. This is particularly useful for serving multiple requests in parallel. Please refer to the example [PACE Multi-instance Inference](../examples/multi_instance_pace.py) for more details on how to set up and use multi-instance inference.
