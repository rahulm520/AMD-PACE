/*******************************************************************************
 * Modifications Copyright (c) 2024 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 *******************************************************************************/

#include <ops/cpu.h>
#include <ops/jit_helper.h>
#include <pace_tensor/pace_aten_interface.h>
#include <utils/utils.h>

namespace pace {

std::vector<float> get_weight_scales_from_aten(const at::Tensor& weight) {
  // Weight scales can be per output channel or per tensor (in cases where the
  // output channel is one), so we need to get the scales for each
  // output channel. It returns a tensor of shape {output_channels}, so we need
  // to convert it to a vector to pass it to the kernel
  TORCH_CHECK(
      qscheme_supported(weight, {at::kPerTensorAffine, at::kPerChannelAffine}),
      "pace::qlinear only supports per channel quantization for weight");

  int num_scale_elements = 1;
  if (weight.qscheme() == at::kPerChannelAffine) {
    num_scale_elements = weight.sizes()[0];
  }

  at::Tensor weight_per_channel_scales =
      at::empty({num_scale_elements}, at::kDouble);
  if (weight.qscheme() == at::kPerTensorAffine) {
    weight_per_channel_scales[0] = at::native::q_scale_quant(weight);
  } else {
    weight_per_channel_scales = at::native::q_per_channel_scales(weight);
  }

  // Convert the weight scales from float64 to float32 (jit backend uses scales
  // in float32) and save it into a vector to compute scales for bias and output
  std::vector<float> weight_scales(weight_per_channel_scales.numel());
  for (int idx = 0; idx < weight_per_channel_scales.numel(); idx++) {
    weight_scales[idx] =
        static_cast<float>(weight_per_channel_scales[idx].item<double>());
  }

  return weight_scales;
}

std::vector<float> get_weight_scales(const at::Tensor& weight) {
  if (weight.has_storage()) {
    return get_weight_scales_from_aten(weight);
  } else {
    return retrieve_pace_tensor_from_dense(weight).get_scales();
  }
}

at::Tensor create_pace_tensor_from_dense(
    const at::Tensor& cpu_tensor,
    transpose_dims trans_dim,
    int ndims) {
  ndims = (ndims != 0) ? ndims : cpu_tensor.dim();
  TORCH_CHECK(
      cpu_tensor.device().is_cpu(),
      "create_pace_tensor_from_dense expects CPU tensor input");
  TORCH_CHECK(
      cpu_tensor.layout() == at::Layout::Strided,
      "create_pace_tensor_from_dense expects strided tensor input");
  TORCH_CHECK(
      ndims == 2,
      "create_pace_tensor_from_dense only supports 2D inputs for now");

  // Create a memory from Aten Tensor
  at::Tensor cpu_tensor_cont = cpu_tensor.contiguous();
  memory zen_mem =
      view_tensor_as_memory(cpu_tensor_cont, trans_dim, ndims, true);

  // Get the weights and @TODO: zero point
  // Assumption is that if the tensor is quantized, it will be a weight tensor
  // and if not, it will be a bias tensor
  std::vector<float> scales;
  if (cpu_tensor.is_quantized()) {
    scales = get_weight_scales(cpu_tensor_cont);
  }

  // Create a PACETensor with memory
  PACETensor pace_tensor = PACETensor(zen_mem, scales, {0});

  // Create a wrapper over the PACETensor and create an Opaque Wrapper over it
  PACETensorWrapperPtr handle =
      c10::make_intrusive<PACETensorWrapper>(std::move(pace_tensor));

  auto dims = cpu_tensor_cont.sizes();
  return at::detail::make_tensor<PACETensorImpl>(
      c10::DispatchKeySet(c10::DispatchKey::CPU),
      scalarTypeToTypeMeta(cpu_tensor_cont.scalar_type()),
      cpu_tensor_cont.options().device(),
      handle,
      std::vector<int64_t>(dims.begin(), dims.end()));
}

PACETensor& retrieve_pace_tensor_from_dense(const at::Tensor& pace_tensor) {
  // @TODO: Make sure it is a pace tensor somehow
  // TORCH_INTERNAL_ASSERT(at::impl::variable_excluded_from_dispatch());
  PACETensorImpl* paceimpl =
      static_cast<PACETensorImpl*>(pace_tensor.unsafeGetTensorImpl());
  return paceimpl->unsafe_opaque_handle()->get_target();
}

memory copy_tensor_into_memory(
    const at::Tensor& atensor,
    memory::dims dims,
    memory::data_type data_type,
    memory::format_tag format) {
  memory::desc zen_memory_desc = memory::desc(dims, data_type, format);

  memory zen_memory = memory(zen_memory_desc, cpu_eng(), JIT_MEMORY_ALLOCATE);
  int num_elements =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());

  int count;
  switch (atensor.scalar_type()) {
    case at::kQInt8: {
      using T = decltype(c10::impl::ScalarTypeToCPPType<at::kQInt8>::t);
      count = num_elements * sizeof(T);
      break;
    }
    case at::kQUInt8: {
      using T = decltype(c10::impl::ScalarTypeToCPPType<at::kQUInt8>::t);
      count = num_elements * sizeof(T);
      break;
    }
    case at::kFloat: {
      using T = decltype(c10::impl::ScalarTypeToCPPType<at::kFloat>::t);
      count = num_elements * sizeof(T);
      break;
    }
    default:
      TORCH_CHECK(
          false, "something occured in optimize::copy_tensor_into_memory");
  }
  std::memcpy(zen_memory.get_data_handle(), atensor.data_ptr(), count);
  return zen_memory;
}

memory view_tensor_as_memory(
    const at::Tensor& atensor,
    memory::dims dims,
    memory::data_type data_type,
    memory::format_tag format) {
  memory::desc zen_memory_desc = memory::desc(dims, data_type, format);
  return memory({zen_memory_desc}, cpu_eng(), atensor.data_ptr());
}

memory view_tensor_as_memory(
    const at::Tensor& atensor,
    transpose_dims trans_dims,
    int ndims,
    bool copy) {
  memory::dims sizes = atensor.sizes().vec();
  if (ndims != 0 && ndims != atensor.dim()) {
    std::vector<int64_t> new_dims(ndims - atensor.dim(), 1);
    new_dims.insert(new_dims.end(), sizes.begin(), sizes.end());
    sizes = new_dims;
  }

  if (trans_dims.dim_a != trans_dims.dim_b) {
    TORCH_CHECK(
        trans_dims.dim_a < sizes.size() && trans_dims.dim_b < sizes.size(),
        "trans_dims is out of range");
    std::swap(sizes[trans_dims.dim_a], sizes[trans_dims.dim_b]);
  }

  tag format = get_default_format(sizes.size(), trans_dims);

  if (copy) {
    return copy_tensor_into_memory(
        atensor, sizes, dtype_to_zen(atensor.scalar_type()), format);
  }

  return view_tensor_as_memory(
      atensor, sizes, dtype_to_zen(atensor.scalar_type()), format);
}

at::Tensor view_memory_as_tensor(
    const memory& mem,
    const at::ScalarType& scalar_type,
    transpose_dims trans_dims) {
  auto mem_desc = mem.get_desc();
#ifdef USE_ONEDNN
  auto dims = mem_desc.get_dims();
#endif
#ifdef USE_ZENDNN
  auto dims = mem_desc.dims();
#endif
  std::vector<int64_t> tensor_dims(dims.begin(), dims.end());

  if (tensor_dims.size() == 2 && trans_dims.dim_a != trans_dims.dim_b) {
    TORCH_CHECK(
        tensor_dims.size() > std::max(trans_dims.dim_a, trans_dims.dim_b),
        "view_memory_as_tensor: trans_dims is out of range for tensor dimensions");
    std::swap(tensor_dims[trans_dims.dim_a], tensor_dims[trans_dims.dim_b]);
  }

  void* data_ptr = mem.get_data_handle();
  auto options = at::TensorOptions().dtype(scalar_type).device(at::kCPU);
  at::Tensor tensor = at::from_blob(data_ptr, tensor_dims, options);
  return tensor;
}
} // namespace pace
