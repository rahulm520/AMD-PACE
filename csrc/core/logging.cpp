/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include <core/logging.h>

namespace pace {

namespace logging {

std::string dtype_to_string(const at::ScalarType& type) {
  switch (type) {
    case at::kByte:
      return "u8";
    case at::kQUInt8:
      return "u8";
    case at::kChar:
      return "s8";
    case at::kQInt8:
      return "s8";
    case at::kInt:
      return "s32";
    case at::kQInt32:
      return "s32";
    case at::kBFloat16:
      return "bf16";
    case at::kFloat:
      return "f32";
    case at::kLong:
      return "s64";
    case at::kQUInt4x2: // 4-bit quantized operator in embedding bag
      return "u4x2";
    default:
      TORCH_CHECK(false, "Please check the dtype!");
  };
}

std::string dtype_to_string(const at::Tensor& atensor) {
  return dtype_to_string(atensor.scalar_type());
}

std::string sizes_to_string(const at::IntArrayRef& atensor_size) {
  std::string sizes_string;
  for (int dim = 0; dim < atensor_size.size(); dim++) {
    sizes_string += std::to_string(atensor_size[dim]);
    if (dim != atensor_size.size() - 1) {
      sizes_string += "x";
    }
  }
  return sizes_string;
}

std::string sizes_to_string(at::Tensor atensor) {
  return sizes_to_string(atensor.sizes());
}

std::string get_current_time() {
  // Get current time as a time_t
  std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
  std::time_t currentTime = std::chrono::system_clock::to_time_t(now);

  // Convert time_t to a tm structure
  std::tm* localTime = std::localtime(&currentTime);

  // Get the milliseconds part
  auto durationSinceEpoch = now.time_since_epoch();
  auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                          durationSinceEpoch) %
      1000;

  // Convert tm structure to a string along with milliseconds
  std::stringstream ss;
  ss << std::put_time(localTime, "%d-%b-%y %H:%M:%S") << "." << std::setw(3)
     << std::setfill('0') << milliseconds.count();
  return ss.str();
}

std::string extra_info_if_quantized(
    const at::Tensor& atensor,
    std::string argname) {
  std::string extra_info;
  if (atensor.is_quantized()) {
    if (atensor.qscheme() == at::kPerTensorAffine) {
      extra_info += " " + argname +
          "_scale:" + std::to_string(atensor.q_scale()) + " " + argname +
          "_zero_point:" + std::to_string(atensor.q_zero_point());
    } else {
      extra_info += " " + argname + "_scale:" + std::to_string(atensor.dim()) +
          " " + argname + "_zero_point:" + std::to_string(atensor.dim());
    }
  }
  return extra_info;
}

std::string extra_info_from_post_ops(
    const at::TensorList& post_ops,
    const std::vector<std::string>& post_ops_algo) {
  std::string post_ops_info;
  if (post_ops.size() > 0) {
    for (int i = 0; i < post_ops.size(); i++) {
      post_ops_info += " attr-post-ops:" + post_ops_algo[i] + " ";
      post_ops_info += sizes_to_string(post_ops[i]);
      post_ops_info +=
          extra_info_if_quantized(post_ops[i], "post_ops_" + std::to_string(i));
    }
  }

  if (post_ops.size() == 0 && post_ops_algo.size() > 0) {
    for (int i = 0; i < post_ops_algo.size(); i++) {
      post_ops_info += " attr-post-ops:" + post_ops_algo[i] + " ";
    }
  }

  return post_ops_info;
}

void TimingLogger::AddInfoBinary(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& output,
    const at::TensorList& post_ops,
    const std::vector<std::string>& post_ops_algo) {
  std::string dtypes = "src0_" + dtype_to_string(a) + " src1_" +
      dtype_to_string(b) + " dst_" + dtype_to_string(output);
  std::string sizes = sizes_to_string(a) + ":" + sizes_to_string(b) + ":" +
      sizes_to_string(output);
  std::string extra_info;
  extra_info += extra_info_if_quantized(a, "src0");
  extra_info += extra_info_if_quantized(b, "src1");
  extra_info += extra_info_if_quantized(output, "dst");
  std::string post_ops_info = extra_info_from_post_ops(post_ops, post_ops_algo);
  {
    std::lock_guard<std::mutex> lock(info_mutex_);
    info_ += dtypes + "," + extra_info + "," + post_ops_info + "," + sizes;
  }
}

void TimingLogger::AddInfoLinear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& output,
    const at::TensorList& post_ops,
    const std::vector<std::string>& post_ops_algo) {
  std::string dtypes = "src_" + dtype_to_string(input) + " wei_" +
      dtype_to_string(weight) + " bia_" + dtype_to_string(bias) + " dst_" +
      dtype_to_string(output);
  std::string sizes = sizes_to_string(input) + ":" + sizes_to_string(weight) +
      ":" + sizes_to_string(bias) + ":" + sizes_to_string(output);
  std::string extra_info;
  extra_info += extra_info_if_quantized(input, "src");
  extra_info += extra_info_if_quantized(weight, "wei");
  extra_info += extra_info_if_quantized(bias, "bia");
  extra_info += extra_info_if_quantized(output, "dst");
  std::string post_ops_info = extra_info_from_post_ops(post_ops, post_ops_algo);
  {
    std::lock_guard<std::mutex> lock(info_mutex_);
    info_ += dtypes + "," + extra_info + "," + post_ops_info + "," + sizes;
  }
}

void TimingLogger::AddInfoEmbedding(
    const at::Tensor& indices,
    const at::Tensor& offsets,
    const at::Tensor& output,
    const at::TensorList& post_ops,
    const std::vector<std::string>& post_ops_algo) {
  std::string dtypes = "indices_" + dtype_to_string(indices) + " offsets_" +
      dtype_to_string(indices) + " dst_" + dtype_to_string(output);
  std::string sizes =
      sizes_to_string(output); // Temporary solution to avoid printing wrong
                               // info for qmerged_embedding_bag_nbit_cat
  std::string extra_info;
  extra_info += extra_info_if_quantized(indices, "ind");
  extra_info += extra_info_if_quantized(offsets, "off");
  extra_info += extra_info_if_quantized(output, "dst");
  std::string post_ops_info = extra_info_from_post_ops(post_ops, post_ops_algo);

  {
    std::lock_guard<std::mutex> lock(info_mutex_);
    info_ += dtypes + "," + extra_info + "," + post_ops_info + "," + sizes;
  }
}

void TimingLogger::AddInfoMlpMlpFusion(
    const at::Tensor& src,
    const std::vector<at::Tensor>& weights1,
    const c10::optional<std::vector<at::Tensor>>& bias1,
    const std::vector<at::Tensor>& weights2,
    const c10::optional<at::Tensor>& bias2,
    std::string nlf,
    const c10::optional<std::vector<at::Tensor>>& weights_gateProj,
    const c10::optional<std::vector<at::Tensor>>& bias_gateProj,
    const at::Tensor& output) {
  std::string dtypes = "src_" + dtype_to_string(src);

  dtypes += (weights_gateProj.has_value() ? " wei_gateProj_" +
                     dtype_to_string(weights_gateProj.value()[0])
                                          : "") +
      (bias_gateProj.has_value()
           ? " bia_gateProj_" + dtype_to_string(bias_gateProj.value()[0])
           : "") +
      " wei1_" + dtype_to_string(weights1[0]) + " bia1_" +
      (bias1.has_value() ? dtype_to_string(bias1.value()[0]) : "none") +
      " wei2_" + dtype_to_string(weights2[0]) + " bia2_" +
      (bias2.has_value() ? dtype_to_string(bias2.value()) : "none") + " dst_" +
      dtype_to_string(output);

  // Proper isolation of sizes for each tensor
  int iterations = weights1.size();
  int weights1_size_0 = weights1[0].sizes()[0] * iterations;
  int weights1_size_1 = weights1[0].sizes()[1];
  std::array<int64_t, 2> wweights1_size = {weights1_size_1, weights1_size_0};
  std::array<int64_t, 2> bbias1_size = {0, 0};
  if (bias1.has_value()) {
    int bias1_size_0 = bias1.value()[0].sizes()[0] * iterations;
    int bias1_size_1 = bias1.value()[0].sizes()[1];
    bbias1_size = {bias1_size_1, bias1_size_0};
  }

  int weights2_size_0 = weights2[0].sizes()[0];
  int weights2_size_1 = weights2[0].sizes()[1] * iterations;
  std::array<int64_t, 2> wweights2_size = {weights2_size_1, weights2_size_0};

  std::string sizes;
  int dst_inter_size_0 = src.sizes()[0];
  int dst_inter_size_1 = weights1_size_0;
  std::array<int64_t, 2> dst_inter_size = {dst_inter_size_0, dst_inter_size_1};

  std::array<int64_t, 2> wweights_gateProj_size = {0, 0};
  if (weights_gateProj.has_value()) {
    int weights_gateProj_size_0 = weights_gateProj.value()[0].sizes()[0];
    int weights_gateProj_size_1 =
        weights_gateProj.value()[0].sizes()[1] * iterations;
    wweights_gateProj_size = {weights_gateProj_size_1, weights_gateProj_size_0};
  }
  std::array<int64_t, 2> bbias_gateProj_size = {0, 0};
  if (bias_gateProj.has_value()) {
    int bias_gateProj_size_0 = bias_gateProj.value()[0].sizes()[0];
    int bias_gateProj_size_1 = bias_gateProj.value()[0].sizes()[1] * iterations;
    bbias_gateProj_size = {bias_gateProj_size_0, bias_gateProj_size_1};
  }
  sizes += sizes_to_string(src) + ":" +
      (weights_gateProj.has_value() ? sizes_to_string(wweights_gateProj_size)
                                    : "") +
      ":" +
      (bias_gateProj.has_value() ? sizes_to_string(bbias_gateProj_size) : "") +
      ":" + sizes_to_string(dst_inter_size);

  sizes += ":" + sizes_to_string(src) + ":" + sizes_to_string(wweights1_size) +
      ":" + (bias1.has_value() ? sizes_to_string(bbias1_size) : "") + ":" +
      sizes_to_string(dst_inter_size) + ":" + sizes_to_string(wweights2_size) +
      ":" + (bias2.has_value() ? sizes_to_string(bias2.value()) : "") + ":" +
      sizes_to_string(output);

  std::string extra_info;
  extra_info += extra_info_if_quantized(src, "src");
  if (weights_gateProj.has_value())
    extra_info +=
        extra_info_if_quantized(weights_gateProj.value()[0], "wei_gateProj");
  if (bias_gateProj.has_value())
    extra_info +=
        extra_info_if_quantized(bias_gateProj.value()[0], "bia_gateProj");
  extra_info += extra_info_if_quantized(weights1[0], "wei1");
  if (bias1.has_value())
    extra_info += extra_info_if_quantized(bias1.value()[0], "bia1");
  extra_info += extra_info_if_quantized(weights2[0], "wei2");
  if (bias2.has_value())
    extra_info += extra_info_if_quantized(bias2.value(), "bia2");
  extra_info += extra_info_if_quantized(output, "dst");

  std::vector<std::string> postops_algo = {nlf};
  std::string post_ops_info = extra_info_from_post_ops({}, postops_algo);

  {
    std::lock_guard<std::mutex> lock(info_mutex_);
    info_ += dtypes + "," + extra_info + "," + post_ops_info + "," + sizes;
  }
}
} // namespace logging

} // namespace pace
