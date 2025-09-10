/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#ifndef UTILS_H
#define UTILS_H

#include <ATen/ATen.h>
#include <algorithm>
#include <vector>

namespace pace {

/**
 * @brief Get the bias if provided, otherwise returns empty tensor
 *
 * @param bias_opt Bias tenosr as an optional value
 * @param options If bias is empty, we use options to extract the return
 * datatype of empty tensor
 * @return at::Tensor
 */
inline at::Tensor get_bias_from_opt(
    const c10::optional<at::Tensor>& bias_opt,
    at::TensorOptions options = c10::TensorOptions()) {
  if (bias_opt.has_value()) {
    c10::MaybeOwned<at::Tensor> bias =
        c10::MaybeOwned<at::Tensor>::borrowed(bias_opt.value());
    return *bias;
  } else {
    return at::empty({}, options);
  }
}
inline std::vector<at::Tensor> get_optional_list_tensors(
    const c10::optional<std::vector<at::Tensor>>& optional_list_tensors) {
  if (optional_list_tensors.has_value()) {
    std::vector<at::Tensor> optional_list_of_tensors =
        optional_list_tensors.value();
    return optional_list_of_tensors;
  } else {
    return std::vector<at::Tensor>();
  }
}
/**
 * @brief Checks if the dtype at type is one of the dtypes
 *
 * @param type
 * @param supported_types
 * @return true
 * @return false
 */
inline bool dtype_supported(
    at::ScalarType type,
    std::vector<at::ScalarType> supported_types) {
  return (
      std::find(supported_types.cbegin(), supported_types.cend(), type) !=
      supported_types.cend());
}

/**
 * @brief Chceks if the tensor has one of the QSchemes
 *
 * @param qtensor
 * @param qschemes
 * @return true
 * @return false
 */
inline bool qscheme_supported(
    at::Tensor qtensor,
    std::vector<at::QScheme> qschemes) {
  return std::find(qschemes.cbegin(), qschemes.cend(), qtensor.qscheme()) !=
      qschemes.cend();
}

/**
 * @brief Check if all the dtypes provided are the same
 *
 * @param types
 * @return true
 * @return false
 */
inline bool is_same_dtype(std::vector<at::ScalarType> types) {
  return (
      std::adjacent_find(types.begin(), types.end(), std::not_equal_to<>()) ==
      types.end());
}

/**
 * @brief Reshape input tensor to 2D for ZenDNN/oneDNN primitives
 *
 * @param input
 * @return input_reshaped
 */
inline at::Tensor flatten_to_2D(at::Tensor input) {
  at::Tensor input_reshaped = input;
  if (input.dim() != 2) {
    const at::SymIntArrayRef input_sizes = input.sym_sizes();
    c10::SymInt flattened_dim = 1;
    for (int64_t i = 0, ndim = input_sizes.size(); i < ndim - 1; ++i) {
      flattened_dim = flattened_dim * input_sizes[i];
    }
    input_reshaped = input.reshape_symint(
        {flattened_dim, input_sizes.at(input_sizes.size() - 1)});
  }
  return input_reshaped;
}

/**
 * @brief Reshape bias tensor to be 2D for ZenDNN/oneDNN primitives
 *
 * @param bias
 * @return reshaped_bias
 */
inline at::Tensor unsqueeze_bias_2D(at::Tensor bias) {
  bias = bias.unsqueeze(0).contiguous();
  return bias;
}

/**
 * @brief Reshape bias tensor vectors individual elements to 2D for
 * ZenDNN/oneDNN primitives
 *
 * @param src
 * @return reshaped_src
 */
inline std::vector<at::Tensor> unsqueeze_bias_2D(std::vector<at::Tensor> bias) {
  for (auto& b : bias) {
    b = b.unsqueeze(0).contiguous();
  }
  return bias;
}

} // namespace pace

#endif // UTILS_H
