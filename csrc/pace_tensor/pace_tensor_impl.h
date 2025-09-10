/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#ifndef PACE_TENSOR_IMPL_H
#define PACE_TENSOR_IMPL_H

#include <ATen/ATen.h>
#include <ops/jit_helper.h>

namespace pace {

using scale_t = std::vector<float>;
using zero_point_t = std::vector<int32_t>;

/**
 * @brief Simple class called PACETensor which is a wrapper over
 * memory with scale and zero point values
 *
 */
class PACETensor {
 private:
  bool optimal_format_ = false;
  memory plain_zen_mem_;
  memory optimized_zen_mem_;

  scale_t scale_;
  zero_point_t zero_point_;

 public:
  PACETensor() = default;
  PACETensor(memory zen_mem, scale_t scale, zero_point_t zero_point)
      : plain_zen_mem_(zen_mem), scale_(scale), zero_point_(zero_point) {}

  // Getter/Setter for memory
  memory get_zen_mem(bool optimal_format = false);
  void set_zen_mem(memory zen_mem, bool optimized = false);

  // returns true if optimized_zen_mem_ is not empty
  bool is_optimized();

  // Getter/Setter for scales
  scale_t get_scales();
  void set_scales(scale_t scale);

  // Getter/Setter for zero points
  zero_point_t get_zero_point();
  void set_zero_point(zero_point_t zero_point);
};

} // namespace pace

#endif // PACE_TENSOR_IMPL_H