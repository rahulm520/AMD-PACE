/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#ifndef QUANTIZE_UTILS_H
#define QUANTIZE_UTILS_H

#include <ATen/ATen.h>
#include <ATen/quantized/Quantizer.h>
#include <ops/cpu.h>
#include <ops/jit_helper.h>
#include <pace_tensor/pace_aten_interface.h>
#include <utils/utils.h>
#include <vector>

namespace pace {

using dt = memory::data_type;
using tag = memory::format_tag;

/**
 * @brief Struct to hold all the scale values for primitive
 *
 */
struct quantize_scales {
  float input_scale;
  float output_scale;
  std::vector<float> weight_scales;
  std::vector<float> zen_bias_scales;
  std::vector<float> zen_output_scales;
};

struct quantize_zeropoint {
  int input_zp;
  int output_zp;
};

/**
 * @brief Get the scale values from all tensors and caluclate
 * the scalues to be used with primitive
 *
 * @param input Input Tenosr
 * @param weight Weight Tenosr
 * @param output_scale Scale value for output
 * @return quantize_scales
 */
inline quantize_scales get_quantize_scales(
    const at::Tensor& input,
    const at::Tensor& weight,
    const float output_scale) {
  // Input scale is per tensor, so we can get it directly
  TORCH_CHECK(
      qscheme_supported(input, {at::kPerTensorAffine}),
      "pace::qlinear only supports per tensor quantization for input");

  quantize_scales q_scales;

  double input_scale = at::native::q_scale_quant(input);
  q_scales.input_scale = input_scale;
  q_scales.output_scale = output_scale;

  // Get the weight scales
  std::vector<float> weight_scales = get_weight_scales(weight);
  q_scales.weight_scales = weight_scales;

  // Compute the bias and output scales required by the kernel
#ifdef USE_ZENDNN
  std::vector<float> zen_bias_scales(weight_scales.size());
  std::vector<float> zen_output_scales(weight_scales.size());
  for (int idx = 0; idx < weight_scales.size(); idx++) {
    zen_bias_scales[idx] = 1 / (weight_scales[idx] * input_scale);
    zen_output_scales[idx] = (weight_scales[idx] * input_scale) / output_scale;
  }
  q_scales.zen_bias_scales = zen_bias_scales;
  q_scales.zen_output_scales = zen_output_scales;
#endif

  return q_scales;
}

} // namespace pace

#endif // QUANTIZE_UTILS_h
