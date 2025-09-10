/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include <core/logging.h>
#include <ops/attention.h>
#include <ops/cpu.h>
#include <ops/jit_helper.h>
#include <ops/kernels/attention_kernels.h>
#include <pace_tensor/pace_aten_interface.h>
#include <torch/library.h>
#include <utils/utils.h>

namespace pace {

at::Tensor attention(
    const at::Tensor& input_Q,
    const at::Tensor& input_K,
    const at::Tensor& input_V,
    const c10::optional<at::Tensor>& input_mask,
    const c10::optional<at::Scalar>& use_KQ) {
  PROFILE_PACE_FUNCTION("attention");

  TORCH_CHECK(
      input_Q.dim() == 4 && input_K.dim() == 4 && input_V.dim() == 4 &&
          input_mask.value().dim() == 4,
      "pace::SDPA attention requires 4D inputs, but recieved: ",
      " input_Q - ",
      input_Q.sizes(),
      ", ",
      " input_K - ",
      input_K.sizes(),
      ", ",
      " input_V - ",
      input_V.sizes(),
      ", and ",
      " input_attention_mask - ",
      input_mask.value().sizes());

  TORCH_CHECK(
      (input_K.sizes() == input_V.sizes()),
      "pace::SDPA attention requires Key and Value sizes to be of "
      "same shape, but recieved: ",
      " input_K - ",
      input_K.sizes(),
      ", ",
      " input_V - ",
      input_V.sizes());

  TORCH_CHECK(
      dtype_supported(input_Q.scalar_type(), {at::kFloat, at::kBFloat16}),
      "pace::SDPA attention only support the dtypes Float and BF16 types for input Query");

  TORCH_CHECK(
      dtype_supported(input_K.scalar_type(), {at::kFloat, at::kBFloat16}),
      "pace::SDPA attention only support the dtypes Float and BF16 types for input Key");

  TORCH_CHECK(
      dtype_supported(input_V.scalar_type(), {at::kFloat, at::kBFloat16}),
      "pace::SDPA attention only support the dtypes Float and BF16 types for input Value");

  TORCH_CHECK(
      dtype_supported(
          input_mask.value().scalar_type(), {at::kFloat, at::kBFloat16}),
      "pace::SDPA attention only support the dtypes Float and BF16 types for input Attention mask");

  memory input_Q_mem = view_tensor_as_memory(input_Q);
  memory input_K_mem = view_tensor_as_memory(input_K);
  memory input_V_mem = view_tensor_as_memory(input_V);

  memory input_mask_mem = memory({}, cpu_eng(), JIT_MEMORY_NONE);
  if (input_mask.has_value())
    input_mask_mem = view_tensor_as_memory(input_mask.value());

  // By default KQ = 0
  int KQ = 0;
  if (use_KQ.has_value())
    KQ = use_KQ.value().toInt();

  TORCH_CHECK(
      (KQ == 0 || KQ == 1),
      "pace::SDPA attention requires use_KQ to be 0 or 1 "
      "but recieved: ",
      " use_KQ - ",
      KQ);

  // Create output tensor memory
  at::Tensor output = at::empty(input_Q.sizes(), input_Q.scalar_type());
  memory output_mem = view_tensor_as_memory(output);

  kernels::attention_kernel(
      input_Q_mem, input_K_mem, input_V_mem, output_mem, input_mask_mem, KQ);

  return output;
}

} // namespace pace

namespace {

// clang-format off
TORCH_LIBRARY_FRAGMENT(pace, m) {

  m.def("attention(Tensor input_Q, Tensor input_K, Tensor input_V, Tensor ? input_mask, Scalar ? use_KQ) -> Tensor", pace::attention);

}
// clang-format on

} // namespace
