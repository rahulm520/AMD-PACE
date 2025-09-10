/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include <core/logging.h>
#include <ops/kernels/libxsmm_linear_kernel.h>
#include <torch/library.h>
#include <utils/utils.h>
#include <mutex>
namespace pace {
// Helper function to validate input and weight tensors
void _validate_inputs(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::string& op_name,
    c10::optional<at::Tensor> multiplier,
    const at::Tensor& bias = at::Tensor()) {
  int expected_input_dim = 3;
  TORCH_CHECK(
      (input.scalar_type() == weight.scalar_type()),
      "pace::",
      op_name,
      " got mismatched types, input: ",
      input.scalar_type(),
      ", weight: ",
      weight.scalar_type(),
      ".");
  TORCH_CHECK(
      dtype_supported(input.scalar_type(), {at::kBFloat16}),
      "pace::",
      op_name,
      " only supports bfloat16 types.");
  TORCH_CHECK(
      input.dim() == expected_input_dim,
      "pace::",
      op_name,
      " expected input to be ",
      expected_input_dim,
      "D, but got ",
      input.dim(),
      "D.");
  TORCH_CHECK( // TODO: dynamically reshape the weight to 5D
      weight.dim() == 5 || weight.dim() == 2,
      "pace::",
      op_name,
      " expected weight to be one of 2D or 5D, but got ",
      weight.dim(),
      "D.");
  if (multiplier.has_value() && multiplier.value().numel() > 0 &&
      multiplier.value().dim() != 0) {
    TORCH_CHECK( // TODO: dynamically reshape the input to 3D
        multiplier.value().dim() == expected_input_dim,
        "pace::",
        op_name,
        " expected input to be ",
        expected_input_dim,
        "D, but got ",
        multiplier.value().dim(),
        "D.");
  }
  if ((bias.numel() > 0 and bias.dim() != 0)) {
    TORCH_CHECK(
        (bias.scalar_type() == input.scalar_type()),
        "pace::",
        op_name,
        " got mismatched types, input: ",
        input.scalar_type(),
        ", bias: ",
        bias.scalar_type(),
        ".");
    TORCH_CHECK(
        bias.dim() == 1,
        "pace::",
        op_name,
        " expected bias to be 1D, but got ",
        bias.dim(),
        "D.");
    TORCH_CHECK(
        bias.size(0) ==
            (weight.dim() == 5 ? weight.size(0) * weight.size(3)
                               : weight.size(0)),
        "pace::",
        op_name,
        " expected bias size to match output size, but got ",
        bias.size(0),
        " and ",
        (weight.dim() == 5 ?: weight.size(0) * weight.size(3), weight.size(0)),
        ".");
  }
}

// Helper function for linear kernal (with and without activation)
at::Tensor _libxsmmlinear(
    at::Tensor& input,
    at::Tensor& weight,
    at::Tensor& bias_opt,
    const std::string& activation,
    c10::optional<at::Tensor> multiplier) {
  PROFILE_PACE_FUNCTION(activation);
  _validate_inputs(input, weight, activation, multiplier, bias_opt);
  at::Tensor output;

  auto multiplier_tensor = multiplier.has_value()
      ? multiplier.value()
      : at::empty({0}, input.options());

  if (weight.dim() == 5) {
    // Existing logic for 5D weight
    auto sizes = input.sizes().vec();
    auto wt_sizes = weight.sizes();
    sizes[sizes.size() - 1] = wt_sizes[0] * wt_sizes[3];
    output = input.new_empty(sizes);

    if (activation == "libxsmmlinear_relu") {
      pace::kernels::libxsmmlinear_kernel<pace::kernels::ReLUActivation>(
          input, multiplier_tensor, weight, bias_opt, output);
    } else if (activation == "libxsmmlinear_gelu") {
      pace::kernels::libxsmmlinear_kernel<pace::kernels::GeluActivation>(
          input, multiplier_tensor, weight, bias_opt, output);
    } else if (activation == "libxsmmlinear_silu") {
      pace::kernels::libxsmmlinear_kernel<pace::kernels::SiLUActivation>(
          input, multiplier_tensor, weight, bias_opt, output);
    } else if (activation == "libxsmmlinear_mul") {
      pace::kernels::libxsmmlinear_kernel<pace::kernels::MulActivation>(
          input, multiplier_tensor, weight, bias_opt, output);
    } else if (activation == "libxsmmlinear_plain") {
      pace::kernels::libxsmmlinear_kernel<pace::kernels::NoOpActivation>(
          input, multiplier_tensor, weight, bias_opt, output);
    } else {
      TORCH_CHECK(false, "Unsupported activation type: ", activation);
    }
  } else {
    // Fallback for non-5D weight
    using namespace logging;
    static std::once_flag log_first_time;
    std::call_once(log_first_time, []() {
      logging::PACE_LOG_WARNING(
          "Using Unoptimized path for libxsmmlinear with 2D weight.");
    });
    weight = weight.transpose(0, 1);
    output = at::matmul(input, weight);
    if ((bias_opt.numel() > 0 and bias_opt.dim() != 0)) {
      output.add_(bias_opt);
    }

    // Apply activation manually
    if (activation == "libxsmmlinear_relu") {
      output = at::relu(output);
    } else if (activation == "libxsmmlinear_gelu") {
      output = at::gelu(output);
    } else if (activation == "libxsmmlinear_silu") {
      output = at::silu(output);
    } else if (activation == "libxsmmlinear_mul") {
      output.mul_(multiplier_tensor);
    } else if (activation != "libxsmmlinear_plain") {
      TORCH_CHECK(false, "Unsupported activation type: ", activation);
    }
  }

  if (activation != "libxsmmlinear_mul") {
    PROFILE_ADD_INFO_LINEAR(input, weight, bias_opt, output, {}, {activation});
  } else {
    PROFILE_ADD_INFO_LINEAR(
        input,
        weight,
        bias_opt,
        output,
        at::ArrayRef({multiplier_tensor}),
        {"mul"});
  }

  return output;
}
//----------------------------------------------------------PUBLIC API

at::Tensor libxsmmlinear_silu(
    at::Tensor& input,
    at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt) {
  auto bias = get_bias_from_opt(bias_opt, input.options());
  return _libxsmmlinear(
      input, weight, bias, "libxsmmlinear_silu", std::nullopt);
}

at::Tensor libxsmmlinear_relu(
    at::Tensor& input,
    at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt) {
  auto bias = get_bias_from_opt(bias_opt, input.options());
  return _libxsmmlinear(
      input, weight, bias, "libxsmmlinear_relu", std::nullopt);
}

at::Tensor libxsmmlinear_gelu(
    at::Tensor& input,
    at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt) {
  auto bias = get_bias_from_opt(bias_opt, input.options());
  return _libxsmmlinear(
      input, weight, bias, "libxsmmlinear_gelu", std::nullopt);
}

at::Tensor libxsmmlinear_plain(
    at::Tensor& input,
    at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt) {
  auto bias = get_bias_from_opt(bias_opt, input.options());
  return _libxsmmlinear(
      input, weight, bias, "libxsmmlinear_plain", std::nullopt);
}

at::Tensor libxsmmlinear_mul(
    at::Tensor& input,
    at::Tensor& multiplier,
    at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt) {
  auto bias = get_bias_from_opt(bias_opt, input.options());
  return _libxsmmlinear(input, weight, bias, "libxsmmlinear_mul", multiplier);
}
} // namespace pace

// Register function with PyTorch
TORCH_LIBRARY_FRAGMENT(pace, m) {
  m.def(
      "libxsmmlinear_plain(Tensor input, Tensor weight, Tensor? bias) -> Tensor",
      pace::libxsmmlinear_plain);
  m.def(
      "libxsmmlinear_gelu(Tensor input, Tensor weight, Tensor? bias) -> Tensor",
      pace::libxsmmlinear_gelu);
  m.def(
      "libxsmmlinear_relu(Tensor input, Tensor weight, Tensor? bias) -> Tensor",
      pace::libxsmmlinear_relu);
  m.def(
      "libxsmmlinear_silu(Tensor input, Tensor weight, Tensor? bias) -> Tensor",
      pace::libxsmmlinear_silu);
  m.def(
      "libxsmmlinear_mul(Tensor input1, Tensor input2, Tensor weight, Tensor? bias) -> Tensor",
      pace::libxsmmlinear_mul);
}
