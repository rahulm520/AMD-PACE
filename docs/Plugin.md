# AMD Platform Aware Compute Engine (AMD PACE)

This sections explains how the AMD PACE is created and the basic idea behind it.

## How it works
The AMD PACE library follows the approach specified in [CUSTOM C++ AND CUDA EXTENSIONS](https://pytorch.org/tutorials/advanced/cpp_extension.html) and [EXTENDING TORCHSCRIPT WITH CUSTOM C++ OPERATORS](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html) in PyTorch documentation. The library is built as a C++ extension to PyTorch and is loaded dynamically by the torch library. The library is built using the `setup.py` script and the `torch.utils.cpp_extension` module in PyTorch.

There are three parts to the library:
1. The PyTorch extension library built using the `setup.py` script.
2. The C++ library built using CMake.
3. The python methods to manipulate the C++ library and provide a high level interface to the user.

### PyTorch extension library
The PyTorch extension library is built using the `setup.py` script. When the library is built, the `setup.py` will create an extension shared object compiled from the C++ sources and link to the methods listed in `csrc/torch_extension_bindings.cpp`. This will be loaded at runtime by the extension library and can be called via `pace._C`. When the python package is installed, these core library methods can be imported from `import pace.core`. For more info on how the core methods are registered and used refer to the [Core Functions documentation](CoreFunctions.md).

### C++ library
The C++ library is built using CMake. The CMake script will build the C++ sources into a shared object library and link them with the extension shared object. The C++ library will link with OneDNN, FBGEMM, libXSMM and TORCH. The ops required will be registered with the torch library using the `TORCH_LIBRARY_FRAGMENT`. For more info on how ops are registered and used refer to the [Ops documentation](Ops.md).

### Python methods
The python methods are used as wrappers for the C++ library and to perform graph transformations. They can also be used to add core functionalities and provide a high level interface to the user. These methods are implemented in `pace` folder. When the python package installed, these methods can be imported from `import pace`. (Under development)
