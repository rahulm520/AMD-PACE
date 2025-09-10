/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include <ops/cpu.h>
#include <ops/kernels/linear_kernel.h>
#include <utils/zen_utils.h>

using tag = memory::format_tag;

namespace pace {

namespace kernels {

void linear_kernel(
    const memory& input,
    PACETensor& pace_weight,
    PACETensor& pace_bias,
    memory& output,
    primitive_attr& attr,
    bool with_reorder,
    const c10::optional<quantize_scales>& q_scales,
    const c10::optional<quantize_zeropoint>& q_zp,
    const std::vector<memory>& post_ops_mem) {
  stream s(cpu_eng());

  std::vector<primitive> net_;
  std::vector<std::unordered_map<int, memory>> net_args_;

  memory weight = pace_weight.get_zen_mem();
  memory bias = pace_bias.get_zen_mem();

  bool update_weight_cache = false;
  bool update_bias_cache = false;

  tag w_b_format = tag::undef;
  if (with_reorder) {
    w_b_format = tag::any;
  }

  bool with_bias = false;
  if (bias.get_desc().get_size() && bias.get_data_handle() != nullptr) {
    with_bias = true;
  }

  memory::desc weight_desc = desc_from_memory(weight, w_b_format);
  memory::desc bias_desc = memory::desc();
  if (with_bias) {
    bias_desc = desc_from_memory(bias, w_b_format);
  }

  // Set scale masks / scales
  if (q_scales.has_value()) {
#ifdef USE_ZENDNN
    attr.set_output_scales(
        op_scale_mask(q_scales->zen_output_scales),
        q_scales->zen_output_scales);
#else // USE_ONEDNN
    attr.set_scales_mask(DNNL_ARG_SRC, op_scale_mask({q_scales->input_scale}));
    attr.set_scales_mask(
        DNNL_ARG_WEIGHTS, op_scale_mask(q_scales->weight_scales));
    attr.set_scales_mask(DNNL_ARG_DST, op_scale_mask({q_scales->output_scale}));
#endif // USE_ONEDNN
  }

  // Set zero points / zero points mask
  if (q_zp.has_value()) {
#ifdef USE_ZENDNN
    attr.set_zero_points(ZENDNN_ARG_SRC, /* mask */ 0, {q_zp->input_zp});
    attr.set_zero_points(ZENDNN_ARG_DST, /* mask */ 0, {q_zp->output_zp});
#else // USE_ONEDNN
    attr.set_zero_points_mask(DNNL_ARG_SRC, /* mask */ 0);
    attr.set_zero_points_mask(DNNL_ARG_DST, /* mask */ 0);
#endif // USE_ONEDNN
  }

#ifdef USE_ZENDNN
  matmul::desc matmul_desc =
      matmul::desc(input.get_desc(), weight_desc, bias_desc, output.get_desc());
  matmul::primitive_desc matmul_prim_desc =
      matmul::primitive_desc(matmul_desc, attr, cpu_eng());
#else // USE_ONEDNN
  matmul::primitive_desc matmul_prim_desc = matmul::primitive_desc(
      cpu_eng(),
      input.get_desc(),
      weight_desc,
      bias_desc,
      output.get_desc(),
      attr);
#endif // USE_ONEDNN

  memory weight_reordered = pace_weight.get_zen_mem(/*optimal_format*/ true);
  // Weight should be reorder if weight is not in the same format
  // as the one expected by the kernel
  // Since weights are already in the INT8 format, we don't need
  // to use the scales/attributes to reorder the weights
  if (matmul_prim_desc.weights_desc() != weight_reordered.get_desc()) {
    weight_reordered = memory(matmul_prim_desc.weights_desc(), cpu_eng());

    net_.push_back(reorder(weight, weight_reordered));
    net_args_.push_back(
        {{JIT_ARG_SRC, weight}, {JIT_ARG_DST, weight_reordered}});
    update_weight_cache = true;
  }

  memory bias_reordered = pace_bias.get_zen_mem(/*optimal_format*/ true);
  // Bias should be reorder if Bias is not empty and
  // is not in the same format as the one expected by the kernel
  // OR bias has a scale that needs to be applied
  bool reorder_bias = with_bias;
  reorder_bias = reorder_bias &&
      ((matmul_prim_desc.bias_desc() != bias_reordered.get_desc()) ||
       q_scales.has_value());
  // This check prevents quantizing the biases again
  reorder_bias = reorder_bias && !pace_bias.is_optimized();
  if (reorder_bias) {
    primitive_attr bias_attr;
    bias_reordered = memory(matmul_prim_desc.bias_desc(), cpu_eng());
    if (q_scales.has_value()) {
#ifdef USE_ZENDNN
      bias_attr.set_output_scales(
          op_scale_mask(q_scales->zen_bias_scales), q_scales->zen_bias_scales);
#endif // USE_ONEDNN
    }
    net_.push_back(reorder(bias, bias_reordered, bias_attr));
    net_args_.push_back({{JIT_ARG_SRC, bias}, {JIT_ARG_DST, bias_reordered}});
    update_bias_cache = true;
  }

  matmul matmul_prim = matmul(matmul_prim_desc);
  std::unordered_map<int, memory> matmul_args;
  // Add the standard args (input, weights, bias? and output)
  matmul_args.insert({JIT_ARG_SRC, input});
  matmul_args.insert({JIT_ARG_WEIGHTS, weight_reordered});
  if (with_bias) {
    matmul_args.insert({JIT_ARG_BIAS, bias_reordered});
  }
  matmul_args.insert({JIT_ARG_DST, output});

#ifdef USE_ONEDNN
  // Add the scale values
  if (q_scales.has_value()) {
    memory input_scale_mem = view_value_as_memory<float>(q_scales->input_scale);
    memory weight_scale_mem =
        view_vector_as_memory<float>(q_scales->weight_scales);
    memory output_scale_mem =
        view_value_as_memory<float>(q_scales->output_scale);

    matmul_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, input_scale_mem});
    matmul_args.insert(
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, weight_scale_mem});
    matmul_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, output_scale_mem});
  }

  // // Add the zero points
  if (q_zp.has_value()) {
    // Create memories from zero points
    memory input_zp_mem = view_value_as_memory<int>(q_zp->input_zp);
    memory output_zp_mem = view_value_as_memory<int>(q_zp->output_zp);
    matmul_args.insert(
        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, input_zp_mem});
    matmul_args.insert(
        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, output_zp_mem});
  }
#endif
  // Add the post ops args if any of them are binary ops
  int post_ops_mem_pos = 0;
  if (post_ops_mem.size() > 0) {
    for (int idx = 0; idx < attr.get_post_ops().len(); idx++) {
      if (attr.get_post_ops().kind(idx) == primitive::kind::binary) {
        matmul_args.insert(
            {{JIT_ARG_ATTR_MULTIPLE_POST_OP(idx) | JIT_ARG_SRC_1,
              post_ops_mem[post_ops_mem_pos++]}});
      }
    }
  }

  net_.push_back(matmul_prim);
  net_args_.push_back(matmul_args);

  assert(net_.size() == net_args_.size() && "Something wrong at exeute");
  for (size_t i = 0; i < net_.size(); ++i) {
    net_.at(i).execute(s, net_args_.at(i));
  }

  // The idea is that if the update_weight_cache / update_bias_cache
  // is true, that means that the desc in the tensor and the new desc
  // does not match. So even if the weight was reordered and saved
  // as optimized once, it could change (especially for BRGEMM kernels)
  if (update_weight_cache) {
    pace_weight.set_zen_mem(weight_reordered, true);
  }

  if (update_bias_cache) {
    pace_bias.set_zen_mem(bias_reordered, true);
  }
}

} // namespace kernels

} // namespace pace
