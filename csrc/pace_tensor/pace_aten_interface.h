/*******************************************************************************
 * Modifications Copyright (c) 2024 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 *******************************************************************************/

#ifndef PACE_ATEN_INTERFACE_H
#define PACE_ATEN_INTERFACE_H

#include <ATen/ATen.h>
#include <ATen/OpaqueTensorImpl.h>
#include <pace_tensor/pace_tensor_impl.h>
#include <utils/zen_utils.h>
#include <vector>

namespace pace {

/**
 * Taken from: aten/src/ATen/native/mkldnn/MKLDNNCommon.cpp
 *
 * `IntrusivePtrTargetWrapper` wraps a custom storage handle  of a tensor
 *  (as template param) and inherits `c10::intrusive_ptr_target` so that it
 *  can be used with `c10::intrusive_ptr`.
 *
 * It currently only supports wrapping the custom handle by:
 * - Constructing with an existing custom handle by copy/move constructor.
 *
 * See `OpaqueTensorImpl::opaque_handle_`.
 *
 * NOTE: if this is generally useful we may want to move this to its own header.
 */
template <typename T>
struct TORCH_API IntrusivePtrTargetWrapper : c10::intrusive_ptr_target {
 private:
  T target_;

 public:
  IntrusivePtrTargetWrapper() = delete;
  IntrusivePtrTargetWrapper(const T& target) : target_(target) {}
  IntrusivePtrTargetWrapper(T&& target) : target_(std::move(target)) {}

  T& get_target() {
    return target_;
  }
};

using PACETensorWrapper = IntrusivePtrTargetWrapper<PACETensor>;
using PACETensorWrapperPtr = c10::intrusive_ptr<PACETensorWrapper>;
using PACETensorImpl = at::OpaqueTensorImpl<PACETensorWrapperPtr>;

/**
 * @brief Get the weight scales from Aten tensor PACETensor
 *
 * @param weight weight Tensor
 * @return std::vector<float>
 */
std::vector<float> get_weight_scales(const at::Tensor& weight);

/**
 * @brief Create a pace tensor from dense object
 *
 * @param atensor Aten Tenosr
 * @param trans_dims Transpose if unequal dim_a and dim_b
 * @param ndims Interpret the Tensor with ndims dimensions
 * @return at::Tensor
 */
at::Tensor create_pace_tensor_from_dense(
    const at::Tensor& cpu_tensor,
    transpose_dims trans_dim = transpose_dims(),
    int ndims = 0);

/**
 * @brief Retrives PACETensor from the Aten Tensor (wrapped as OpaqueTensor)
 *
 * @param mem_tensor Aten Tenosr
 * @return PACETensor&
 */
PACETensor& retrieve_pace_tensor_from_dense(const at::Tensor& mem_tensor);

/**
 * @brief Use the data pointer from the Aten Tenosr to create a memory
 *
 * @param atensor Aten Tensor which contains the data
 * @param trans_dims Transpose if unequal dim_a and dim_b
 * @param ndims Interpret the Tensor with ndims dimensions
 * @return memory
 */
memory view_tensor_as_memory(
    const at::Tensor& atensor,
    transpose_dims trans_dim = transpose_dims(),
    int ndims = 0,
    bool copy = false);

/**
 * @brief Use the data pointer from the JIT memory to create a Aten Tensor
 *
 * @param mem Memory which contains the data
 * @param scalar_type The type of the memory
 * @param trans_dims Transpose if unequal dim_a and dim_b
 * @return memory
 */
at::Tensor view_memory_as_tensor(
    const memory& mem,
    const at::ScalarType& scalar_type,
    transpose_dims trans_dim = transpose_dims());
} // namespace pace

#endif // PACE_ATEN_INTERFACE_H
