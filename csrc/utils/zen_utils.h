/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#ifndef ZEN_UTILS_H
#define ZEN_UTILS_H

#include <ATen/ATen.h>
#include <ops/cpu.h>
#include <ops/jit_helper.h>
#include <vector>

namespace pace {

using dt = memory::data_type;
using tag = memory::format_tag;

/**
 * @brief Converts the Aten dtype to supported memory::data_type
 *
 * @param atensor_type
 * @return memory::data_type
 */
inline memory::data_type dtype_to_zen(at::ScalarType atensor_type) {
  switch (atensor_type) {
    case at::kByte:
      return dt::u8;
    case at::kQUInt8:
      return dt::u8;
    case at::kChar:
      return dt::s8;
    case at::kQInt8:
      return dt::s8;
    case at::kInt:
      return dt::s32;
    case at::kQInt32:
      return dt::s32;
    case at::kBFloat16:
      return dt::bf16;
    case at::kFloat:
      return dt::f32;
    default:
      TORCH_CHECK(false, "pace does not support the dtype at the moment!");
  };
}

/**
 * @brief Returns the C++ data type in memory data type format.
 *
 * @tparam T C++ dtype
 * @return memory::data_type
 */
template <typename T>
inline memory::data_type dtype_to_zen() {
  if constexpr (std::is_same<T, uint8_t>::value)
    return dt::u8;
  else if constexpr (std::is_same<T, int8_t>::value)
    return dt::s8;
  else if constexpr (
      std::is_same<T, int32_t>::value || std::is_same<T, int>::value)
    return dt::s32;
  else if constexpr (std::is_same<T, float>::value)
    return dt::f32;
  else
    return dt::undef;
}

struct transpose_dims {
  int dim_a;
  int dim_b;

  transpose_dims(int dim_a, int dim_b) : dim_a(dim_a), dim_b(dim_b) {}
  transpose_dims() : dim_a(0), dim_b(0) {}
};

/**
 * @brief Get the default format for the dimension provided
 *
 * @param adims number of dimensions
 * @param transpose_dims Dimensions to be transposed
 * @return memory::format_tag
 */
inline memory::format_tag get_default_format(
    const int& adims,
    const transpose_dims& trans_dims = transpose_dims()) {
  TORCH_CHECK(
      trans_dims.dim_a >= 0 && trans_dims.dim_a < adims &&
          trans_dims.dim_b >= 0 && trans_dims.dim_b < adims,
      "Invalid transpose_dims recieved for get_default_format!");
  TORCH_CHECK(
      adims > 0 && adims <= 4,
      "Invalid dimensions recieved for get_default_format!");

  // If the dimensions are not provided, we assume no transpose
  // If the dimensions are provided, we assume transpose
  bool transpose = (trans_dims.dim_a != trans_dims.dim_b);

  if (!transpose) {
    switch (adims) {
      case 1:
        return tag::a;
      case 2:
        return tag::ab;
      case 3:
        return tag::abc;
      case 4:
        return tag::abcd;
      default:
        TORCH_CHECK(false, "Invalid dimensions for get_default_format!");
    }
  } else {
    switch (adims) {
      case 2:
        return tag::ba;
      case 3:
        TORCH_CHECK(
            trans_dims.dim_a == 1 && trans_dims.dim_b == 2,
            "The transposed format is not accepted within PACE!");
        return tag::acb;
      case 4:
        TORCH_CHECK(
            trans_dims.dim_a == 2 && trans_dims.dim_b == 3,
            "The transposed format is not accepted within PACE!");
        return tag::abdc;
      default:
        TORCH_CHECK(false, "Invalid dimensions for get_default_format!");
    }
  }
}

/**
 * @brief Count the number of specific post-ops in the post_ops struct
 *
 * @param post_ops
 * @param kind
 * @return int
 */
inline int count_post_ops(const post_ops post_ops, primitive::kind kind) {
  int count = 0;
  for (int idx = 0; idx < post_ops.len(); idx++) {
    if (post_ops.kind(idx) == kind) {
      count++;
    }
  }
  return count;
}

/**
 * @brief Get the scale mask for operation
 * If the scale is scalar mask is 1, otherwise 2
 *
 * @param scale
 * @return int
 */
inline int op_scale_mask(const std::vector<float> scale) {
  return (scale.size() > 1) ? 2 : 0;
}

/**
 * @brief Get the descriptor of of the memory
 *
 * @param zen_mem
 * @param format
 * @return memory::desc
 */
inline memory::desc desc_from_memory(
    const memory zen_mem,
    tag format = tag::undef) {
  memory::desc desc = zen_mem.get_desc();
  if (format != tag::undef) {
#ifdef USE_ZENDNN
    desc = memory::desc(desc.dims(), desc.data_type(), format);
#else
    desc = memory::desc(desc.get_dims(), desc.get_data_type(), format);
#endif
  }
  return desc;
}

/**
 * @brief Overload for view_tensor_as_memory
 * This function creates a memory object from a tensor
 * The shape needs to be provided as memory::dims
 *
 * @tparam T data type of the tensor
 * @param vec vector of data
 * @param shape shape provided as memory::dims
 * @param dtype memory::data_type to be used (If not provided, it is inferred
 * from T)
 * @return memory memory object
 */
template <typename T>
memory view_vector_as_memory(
    const std::vector<T>& vec,
    const memory::dims& shape,
    const memory::data_type& dtype = memory::data_type::undef) {
  memory::data_type jit_dtype;
  if (dtype != dt::undef) {
    jit_dtype = dtype;
  } else {
    jit_dtype = dtype_to_zen<T>();
  }

  if (jit_dtype == memory::data_type::undef) {
    TORCH_CHECK(false, "view_vector_as_memory: Unsupported data type");
  }

  memory::desc zen_memory_desc =
      memory::desc(shape, jit_dtype, get_default_format(shape.size()));
  memory zen_memory =
      memory(zen_memory_desc, cpu_eng(), const_cast<T*>(vec.data()));

  return zen_memory;
}

/**
 * @brief Overload for view_vector_as_memory
 * This function creates a memory object from a vector of data
 * The shape of the memory object is inferred from the vector
 *
 * @tparam T data type of the vector
 * @param vec vector of data
 * @param dtype memory::data_type to be used (If not provided, it is inferred
 * from T)
 * @return memory memory object
 */
template <typename T>
memory view_vector_as_memory(
    const std::vector<T>& vec,
    const memory::data_type& dtype = memory::data_type::undef) {
  memory::dims shape = {static_cast<int64_t>(vec.size())};
  return view_vector_as_memory<T>(vec, shape, dtype);
}

/**
 * @brief This function creates a memory object from a single value
 *
 * @tparam T data type of the vector
 * @param value single value of type T
 * @param dtype memory::data_type to be used (If not provided, it is inferred
 * from T)
 * @return memory memory object
 */
template <typename T>
memory view_value_as_memory(
    const T& value,
    const memory::data_type& dtype = memory::data_type::undef) {
  memory::data_type jit_dtype;
  if (dtype != dt::undef) {
    jit_dtype = dtype;
  } else {
    jit_dtype = dtype_to_zen<T>();
  }

  if (jit_dtype == memory::data_type::undef) {
    TORCH_CHECK(false, "view_value_as_memory: Unsupported data type");
  }

  memory::desc zen_memory_desc = memory::desc({1}, jit_dtype, tag::a);
  memory zen_memory =
      memory(zen_memory_desc, cpu_eng(), const_cast<T*>(&value));

  return zen_memory;
}

} // namespace pace

#endif // ZEN_UTILS_H