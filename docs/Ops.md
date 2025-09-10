# Ops supported by AMD PACE

The ops in AMD PACE are implemented as PyTorch C++ extensions with [TORCH_LIBRARY_FRAGMENT](https://pytorch.org/cppdocs/api/define_library_8h_1a2eabc7781e58671237d9d0d282ee1814.html). The ops are defined in `csrc/ops`.

### Using the Ops:
To load/use these ops/kernels, load the `torch` and `pace` libraries as follows:
```
import torch
import pace
```

This will dynamically link the ops registered in the `pace` library to the `torch` library. Ops defined in the AMD PACE are listed below.

1. [Linear Ops](#linear-ops)
2. [Embedding Bag Ops](#embedding-ops)
3. [Binary Ops](#binary-ops)
4. [mlp_mlp_fusion](#mlp_mlp_fusion-ops)

# Linear Ops
These Op implements an inner product of input and weight matrices. The input is a 2D tensor of shape `[batch_size, input_features]` and the weight matrix is a 2D tensor of shape `[output_features, input_features]`. Optionally a bias tensor of shape `[output_features]` can be passed. The output is a 2D matrix of shape `[batch_size, output_features]`. The Op is implemented using ZenDNN/OneDNN primitive: `matmul`.

1. [linear](#linear)
2. [linear_relu](#linear_relu)
3. [qlinear](#qlinear)
4. [qlinear_relu](#qlinear_relu)
5. [qlinear_mul_add](#qlinear_mul_add)
6. [qlinear_sigmod](#qlinear_sigmod)


### linear
* Operation: `torch.ops.pace.linear`
* Graph node type: `pace::linear`
* PostOps: None
* Input Types Supported: FP32/BF16
* Weight Types Supported: FP32/BF16
* Bias Types Supported: FP32/BF16
* Output Types Supported: FP32/BF16
* Arguments:
    * `input`: Input tensor of shape ND `[batch_size, ..., input_features]`.
    * `weight`: Weight tensor of shape 2D `[output_features, input_features]`.
    * `bias`: Bias tensor of shape 1D `[output_features]`.
* File: `csrc/ops/kernels/linear.cpp`
* Correctness Verified: Yes
* Note: All the input, weight and bias tensors must be of the same type.

### linear_relu
* Operation: `torch.ops.pace.linear_relu`
* Graph node type: `pace::linear_relu`
* PostOps: ReLU
* Input Types Supported: FP32/BF16
* Weight Types Supported: FP32/BF16
* Bias Types Supported: FP32/BF16
* Output Types Supported: FP32/BF16
* Arguments:
    * `input`: Input tensor of shape ND `[batch_size, ..., input_features]`.
    * `weight`: Weight tensor of shape 2D `[output_features, input_features]`.
    * `bias`: Bias tensor of shape 1D `[output_features]`.
* File: `csrc/ops/kernels/linear.cpp`
* Correctness Verified: Yes
* Note: All the input, weight and bias tensors must be of the same type.

### qlinear
* Operation: `torch.ops.pace.qlinear`
* Graph node type: `pace::qlinear`
* PostOps: None
* Input Types Supported: QUINT8/QINT8
* Weight Types Supported: QINT8
* Bias Types Supported: FP32/INT32
* Output Types Supported: FP32/INT8
* Arguments:
    * `input`: Input tensor of shape ND `[batch_size, ..., input_features]`.
    * `weight`: Weight tensor of shape 2D `[output_features, input_features]`.
    * `bias`: Bias tensor of shape 1D `[output_features]`.
    * `output_scale`: Output scale of dtype double.
    * `output_zero_point`: Output zero point of dtype int.
    * `output_dtype`: Output dtype of the tensor of type torch.dtype. For FP32 output, provide `output_scale` as `1.0` and  `output_zero_point` as `0`.
* File: `csrc/ops/kernels/linear.cpp`

### qlinear_relu
* Operation: `torch.ops.pace.qlinear_relu`
* Graph node type: `pace::qlinear_relu`
* PostOps: ReLU
* Input Types Supported: QUINT8/QINT8
* Weight Types Supported: QINT8
* Bias Types Supported: FP32/INT32
* Output Types Supported: FP32/INT8
* Arguments:
    * `input`: Input tensor of shape ND `[batch_size, ..., input_features]`.
    * `weight`: Weight tensor of shape 2D `[output_features, input_features]`.
    * `bias`: Bias tensor of shape 1D `[output_features]`.
    * `output_scale`: Output scale of dtype double.
    * `output_zero_point`: Output zero point of dtype int.
    * `output_dtype`: Output dtype of the tensor of type torch.dtype. For FP32 output, provide `output_scale` as `1.0` and  `output_zero_point` as `0`.
* File: `csrc/ops/kernels/linear.cpp`


### qlinear_mul_add
* Operation: `torch.ops.pace.qlinear_mul_add`
* Graph node type: `pace::qlinear_mul_add`
* PostOps: Mul -> Add
* Input Types Supported: QUINT8/QINT8
* Weight Types Supported: QINT8
* Bias Types Supported: FP32/INT32
* Output Types Supported: FP32
* Arguments:
    * `input`: Input tensor of shape ND `[batch_size, ..., input_features]`.
    * `weight`: Weight tensor of shape 2D `[output_features, input_features]`.
    * `bias`: Bias tensor of shape 1D `[output_features]`.
    * `multiplier`: Multiplier tensor of shape ND `[batch_size, ..., input_features]`.
    * `addend`: Addend tensor of shape ND `[batch_size, ..., input_features]`.
    * `alpha`: Alpha for the addend of type float. Only 1 is supported for now.
* File: `csrc/ops/kernels/linear.cpp`

### qlinear_sigmod
* Operation: `torch.ops.pace.qlinear_sigmod`
* Graph node type: `pace::qlinear_sigmod`
* PostOps: Sigmoid
* Input Types Supported: QUINT8/QINT8
* Weight Types Supported: QINT8
* Bias Types Supported: FP32/INT32
* Output Types Supported: FP32
* Arguments:
    * `input`: Input tensor of shape ND `[batch_size, ..., input_features]`.
    * `weight`: Weight tensor of shape 2D `[output_features, input_features]`.
    * `bias`: Bias tensor of shape 1D `[output_features]`.
* File: `csrc/ops/kernels/linear.cpp`

# Embedding Ops
The embedding bag ops

1. [qmerged_embedding_bag_nbit_cat](#qmerged_embedding_bag_nbit_cat)

### qmerged_embedding_bag_nbit_cat
*Note: This method is to be not called directly, this is to be used within AMD PACE if it finds the appropriate pattern.*
* Operation: `torch.ops.pace.qmerged_embedding_bag_nbit_cat`
* Graph node type: `pace::qmerged_embedding_bag_nbit_cat`
* PostOps: Implicit Concat
* Index Type Supported: INT64/INT32
* Offset Type Supported: INT64/INT32
* Weight Types Supported: QINT8/QINT4x2
* Output Types Supported: FP32
* Arguments:
    * `weights`: A vector of size `num_tables` with type `EmbeddingPackedParamsBase`.
    * `indices`: Indices tensor of shape `[batch_size, num_indices]`.
    * `offsets`: Indices tensor of shape `[batch_size + 1, num_indices]`.
    * `dense`: Dense input tensor to be concatenated with the embedding output of shape `[batch_size, embedding_dim]`.
    * `bit_width`: Bit width of weights. Can either be `8` or `4`.
* File: `csrc/ops/kernels/embedding_bag.cpp`
* Extended operator from [PyTorch implementation](https://github.com/pytorch/pytorch/blob/v2.2.0/aten/src/ATen/native/quantized/cpu/qembeddingbag.cpp#L396).

# Binary Ops
The binary ops

1. [qmul_add](#qmul_add)

### qmul_add
*Note: This method is to be not called directly, this is to be used within AMD PACE if it finds the appropriate pattern.*

This operator implements fused qmul -> qadd. It can take it combination of inputs as mentioned below. This operator implements a special connection in DLRMv2 model to improve accuracy without compromising on performance.
* Operation: `torch.ops.pace.qmul_add`
* Graph node type: `pace::qmul_add`
* Multiplier Type Supported: FP32
* Multiplicand Type Supported: INT8
* Addend Types Supported: INT8/FP32
* Output Types Supported: INT8
* Arguments:
    * `a`: Tensor of shape `MxN` where 96 is a factor of N.
    * `b`: Tensor of shape `MxN` where 96 is a factor of N.
    * `addend`: Tensor of shape `MxN` where 96 is a factor of N.
    * `o_scale`: Output scale of dtype double.
    * `o_zero_point`: Output zero point of dtype int.
    * `o_dtype`: Output dtype of the tensor of type torch.dtype.
* File: `csrc/ops/kernels/binary.cpp`


### mlp_mlp_fusion
*Note: This method is to be not called directly, this is to be used within AMD PACE if it finds the appropriate pattern.*
This operator implements fused linear + linear for the FFN layer of transformers. It can currently take in the following formats of inputs. This operator implements the IMBPS mlp flow to control the intermediate buffer memory between the two mlp's. Effectively creating the two as a joint operation at cache level.(similar to what postops do at a register level) refer the IMBPS poster for more info.

* Operation it aims to replace LLamaMLP/ OPTDecoderMLP(not a specific function) in HF/vLLM. :  `torch.ops.pace.mlp_mlp_fusion`
* Data Type support configs(src_f32/bf16 weightsMLP1_f32/bf16 inter_activation_f32/bf16 weightsMLP2_f32/bf16 final_activation_f32/bf16)
* Arguments:
    * `src` Src tensor to the first matmul opearation
    * `weight` Array of weight tensors to the first matmul opearation
    * `bias` Array of Bias tensors to be added to first matmul operation
    * `weights2` Array of weights tensors of the second matmul operation
    * `bias2` Bias tensor to the second matmul operation
    * `nlf` Non linearity function
    * `weights_gateProj` Array of weight tensors to the gate projection operation
    * `bias_gateProj` Array of Bias tensors to be added to gate projection operation

* File: `csrc/ops/kernels/mlp_kernel.cpp`