# How to contribute to AMD PACE

This document provides guidelines and instructions for contributing to the AMD PACE library. It covers various aspects of development, including adding external libraries, creating operators, creating core functions, logging, and basic developer checks.

## Table of Contents
1. [Adding an external library](#adding-an-external-library)
2. [Creating a Operator](#creating-a-operator)
3. [Creating a Python Operator](#creating-a-python-operator)
4. [Creating a core function](#creating-a-core-function)
5. [Logging in AMD PACE](#logging-in-amd-pace)
6. [Basic Developer Checks](#basic-developer-checks)

## Adding an external library
The implementation for building and linking with external libraries are under `cmake/Build{Library}.cmake`. All the libraries are downloaded and build using the `ExternalProject_Add` command in CMake. For more information on `ExternalProject_Add`, please refer to the [CMake documentation](https://cmake.org/cmake/help/latest/module/ExternalProject.html). The external libraries are built as static objects and linked with the AMD PACE library.

Once the library is built, set `{Library}_INCLUDE_DIR` and `{Library}_STATIC_LIB` in the cmake file itself. This is then used to link the library with the AMD PACE library in `csrc/CMakeLists.txt`. If there is any specific order in which the libraries need to be linked, please follow the order in the `csrc/CMakeLists.txt` file.

## Creating a Operator
All the operators are implemented under `csrc/ops/` directory. All the operators follow the same structure and naming conventions. To create a new operator in AMD PACE, follow the steps below:
1. Create a new file under `csrc/ops/` with the name of the operator. For example, create files `csrc/ops/opname.h` `csrc/ops/opname.cpp`, `csrc/ops/kernels/opname_kernel.h` and `csrc/ops/kernels/opname_kernel.cpp`.
The directory structure should look like:
    ```
    csrc
    ├── ops
    ├── kernels
    │   ├── [Optional] opname_kernel_avx512.cpp
    │   ├── opname_kernel.cpp
    │   └── opname_kernel.h
    ├── opname.cpp
    └── opname.h
    ```

2. Define the op and the kernel as follows:
    1. Use the namespace `pace` to define the operator. The operator should be declared in the `opname.h` file.
    2. Use the namespace `pace::kernels` to define the kernel and the kernel should be declared in the `opname_kernel.h` file, should be called from the op implementation in the `opname.cpp` file. This file is ideally used to make any safety checks and to make sure the inputs and outputs are valid and for any pre-processing or post-processing of the inputs and outputs.
    3. Any kernel implementation within AMD PACE should be under the `pace::kernels::impl` namespace. This will help to identify and keep track of the kernel implementations vs the redirections into external libraries.
    4. If the kernel is specifically AVX512 based, then the kernel should be defined in `opname_kernel_avx512.cpp`. The kernel should still be declared in the `opname_kernel.h` file as a declaration. If not, the kernel should be defined in `opname_kernel.cpp` itself.

    For example,

    * `opname.h`:
        ```cpp
        namespace pace {
            // Op declaration
            at::Tensor opname(...);
        }
        ```
    * `opname.cpp`:
        ```cpp
        namespace pace {
            // Op definition
            at::Tensor opname(...) {
                // Op implementation
                opname_kernel(...);
            }
        }
        ```
    * `opname_kernel.h`:
        ```cpp
        namespace pace::kernels {
            // Kernel declaration
            void opname_kernel(...);
        }
        ```
    * `opname_kernel.cpp`:
        ```cpp
        namespace pace::kernels {
            // Kernel definition
            void opname_kernel(...) {
                // Kernel implementation
                // should redirect to AVX512 kernel if available
            }
        }
        ```
    The method names can be anything as long as they make sense and are consistent with the naming conventions.

3. All operators should have a profiling mechanism and logging mechanism enabled. There is a logging and timing module already present in the file: `csrc/core/logging.h` and can be invoked by including the file in the operator file. The logging and timing module should be used to log the input and output shapes, the time taken for the operation, and any other relevant information. The logging and timing module should be used as follows:
    ```cpp
    #include "core/logging.h"

    namespace pace {
        at::Tensor opname(...) {
            // Start the timer
            // The name of the method and the name given to the timer should be the same
            PROFILE_PACE_FUNCTION("opname");
            // Operator implementation
            ...
            // Log the input and output shapes
            PROFILE_ADD_INFO(...)
        }
    }
    ```
    Some macros are available based on the type of operation such as linear, binary etc. to make it easier to log the input and output shapes. The macros are defined in `csrc/core/logging.h` file.

    The timer works on the logic of scoping. When `PROFILE_PACE_FUNCTION("opname")` is called, the timer starts and when the scope ends, the timer stops and logs the time taken for the operation. The timer is thread safe and can be used in multi-threaded environments.

3. Once the op is defined and implementation is complete, within the `opname.cpp` file, you can register the op with the torch library as follows:
    ```cpp
    TORCH_LIBRARY_FRAGEMENT(pace, m) {
        m.def(
            "operator_name(operator_schema)",
            &pace::opname
        );
    }
    ```

4. The function can be imported and used in the python code as follows:
    ```python
    # Make sure to import torch before importing pace
    import torch
    import pace

    ret = torch.ops.pace.opname(...)
    ```

5. Once the method is implemented, it needs to be documented in the `docs/Ops.md` file. The documentation should include the method signature, the input and output types, and a brief description of the operator.

> For a complete example of an operator, refer to the binary operator in `csrc/ops/binary.h` and `csrc/ops/binary.cpp` for operator implementation and `csrc/ops/kernels/binary_kernel.h`, `csrc/ops/kernels/binary_kernel.cpp`, and `csrc/ops/kernels/binary_kernel_avx512.cpp` for kernel implementation.

### Note:
1. Make sure that the op is registered outside of the `pace` namespace so that the op can be loaded dynamically by the torch library.
2. The AVX512 kernel should go in the file `opname_kernel_avx512.cpp` only as only those files are compiled with the AVX512 flags. Failing to do so might result in errors during compilation.
3. All AVX512 kernels should have a reference implementation in the `opname_kernel.cpp` file. This is required for the fallback mechanism in case the AVX512 kernel is not supported on the target machine and for testing purposes.

## Creating a Python Operator
Please refer to [PythonOps.md](PythonOps.md#adding-new-operators-and-backends) for more details on how to create a Python operator.


## Creating a core function
All the core functions are implemented under `csrc/core/` directory. All the core functions follow the same structure and naming conventions. To create a new core function in AMD PACE, follow the steps below:
1. Create a new file under `csrc/core/` with the name of the function. For example, create files `csrc/core/core_method.h` and `csrc/core/core_method.cpp`.
The directory structure should look like:
    ```
    core/
    ├── core_method.cpp
    └── core_method.h
    ```
2. Define the core function as follows:

    1. Use the namespace `pace` to define the function. The function should be declared in the `core_method.h` file.
    2. The function should be defined in the `core_method.cpp` file.
    For example,
    * `core_method.h`:
        ```cpp
        namespace pace {
            // Function declaration
            [return type] core_method(...);
        }
        ```
    * `core_method.cpp`:
        ```cpp
        namespace pace {
            // Function definition
            [return type]  core_method(...) {
                // Function implementation
            }
        }
        ```

3. Once the function is defined and implementation is complete, within the `core_method.cpp` file, you can register the function with the AMD PACE library as follows:
    1. Import the function in the `csrc/torch_extension_bindings.cpp` file.
    2. Register the function within `torch_extension_bindings` method:
        ```cpp
        m.def(
            "core_method",
            &pace::core_method
        );
        ```
4. The function can be imported and used in the python code as follows:
    ```python
    # Make sure to import torch before importing pace
    import torch
    import pace

    ret = pace.core.core_method(...)
    ```
5. Once the method is implemented, it needs to be documented in the `docs/CoreFunctions.md` file. The documentation should include the method signature, the input and output types, and a brief description of the function.


## Logging in AMD PACE
There is a logging module that is available in AMD PACE to be used with both C++ and Python. The logging module is available in the `csrc/core/logging.h` file. There are 5 levels of logging available in AMD PACE -> `DEBUG`, `PROFILE`,  `INFO`, `WARNING`, `ERROR`.

The logging can be controlled by setting the environment variable `PACE_LOG_LEVEL`. Refer to [README](../README.md#verbose) for more details.

To make use of Logger in C++, include the `logging.h` file in the file where you want to log the information, and can be called using the following macros:

```cpp
#include "core/logging.h"
PACE_LOG_DEBUG(...)`: // Used to log debug information.
PACE_LOG_PROFILE(...)`: // Used to log profiling information.
PACE_LOG_INFO(...)`: // Used to log information.
PACE_LOG_WARNING(...)`: // Used to log warnings.
PACE_LOG_ERROR(...)`: // Used to log errors.
```

The logging module in Python is a wrapper around the C++ logging module. The logging module in Python is available in the `pace.utils.logging` module. The logging module in Python has the same levels as the C++ logging module. The logging module in Python can be used as follows. To make use of the different logging levels, the logger should be initialized as follows:
```python
from pace.utils.logging import pacelogger, logLevel

pacelogger(logLevel.DEBUG, "...") # DEBUG level
pacelogger(logLevel.PROFILE, "...") # PROFILE level
pacelogger(logLevel.INFO, "...") # INFO level
pacelogger(logLevel.WARNING, "...") # WARNING level
pacelogger(logLevel.ERROR, "...") # ERROR level
```

For LLM modules, this is abstracted one more level. This is to capture some extra information(please make sure to use these methods inside any python LLM related implementations) , and can be accessed like such:
```python
from pace.utils.logging import PACE_LLM_DEBUG, PACE_LLM_INFO, PACE_LLM_WARNING, PACE_LLM_ASSERT

PACE_LLM_DEBUG("...")
PACE_LLM_INFO("...")
PACE_LLM_WARNING("...")
PACE_LLM_ASSERT(CONDITION, "...")
```
`PACE_LLM_ASSERT` is a special case, where it will raise an exception if the condition is not met. If condition is not met, it will raise an assertion error with the message logged.

## Basic Developer Checks
Before raising patches for review make sure of the following:

> Currently basic linting/formatting is only available. Later more linting will be enforced.

1. Code styling for C++: All the C++ files within AMD PACE follows the PyTorch format of formatting. The formatting module is integrated into AMD PACE itself. To make use of it, while building the extension, use the flag `ENABLE_CLANG_FORMAT`.

    ```shell
    ENABLE_CLANG_FORMAT=ON python setup.py [install|bdist_wheel]
    ```

2. Code styling for Python: All the Python files within AMD PACE follows the standard `black` formatting and can be invoked as follows:
    ```
    black .
    ```
    This will format all the python files within the directory.

    Once you invoke `black`, run `flake8`
    ```
    flake8 .
    ```
    It will check for any linting errors in the code. Make sure to fix all the errors before raising a PR.
    > NOTE: Once you run `flake8`, make sure to run `black` again as `flake8` might change the formatting of the code.
3. Make sure that the library builds and runs with some basic examples. Unit tests also need to be added for methods exposed.
