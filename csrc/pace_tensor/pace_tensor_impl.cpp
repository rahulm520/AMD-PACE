/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include <ATen/OpaqueTensorImpl.h>
#include <ops/cpu.h>
#include <pace_tensor/pace_tensor_impl.h>

namespace pace {

// Return the memory
memory PACETensor::get_zen_mem(bool optimal_format) {
  if (optimal_format && is_optimized()) {
    return optimized_zen_mem_;
  }
  return plain_zen_mem_;
}

// Set the memory in the tensor
void PACETensor::set_zen_mem(memory zen_mem, bool optimal_format) {
  if (optimal_format) {
    optimized_zen_mem_ = zen_mem;
  } else {
    plain_zen_mem_ = zen_mem;
  }
  optimal_format_ = optimal_format;
}

// Returns the value of optimal_format
bool PACETensor::is_optimized() {
  return (optimal_format_ && optimized_zen_mem_.get_data_handle() != nullptr);
}

// Get the scales
scale_t PACETensor::get_scales() {
  return scale_;
}

// Set the scales
void PACETensor::set_scales(scale_t scale) {
  scale_.resize(scale.size());
  std::copy(scale.begin(), scale.end(), scale_.begin());
}

// Get the zero points
zero_point_t PACETensor::get_zero_point() {
  return zero_point_;
}

// Set the zero points
void PACETensor::set_zero_point(zero_point_t zero_point) {
  zero_point_.resize(zero_point.size());
  std::copy(zero_point.begin(), zero_point.end(), zero_point_.begin());
}

} // namespace pace
