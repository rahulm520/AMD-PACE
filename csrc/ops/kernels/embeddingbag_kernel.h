/*******************************************************************************
 * Modifications Copyright (c) 2024 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 * Portions of this file consist of AI-generated content
 *******************************************************************************/

#ifndef EMBEDDINGBAG_KERNEL_H
#define EMBEDDINGBAG_KERNEL_H

#include <ATen/ATen.h>

namespace pace {

namespace kernels {

/**
 * @brief Modified version from
 * aten/src/ATen/native/quantized/cpu/qembeddingbag.cpp:embedding_bag_4bit_helper()
 * Had to make a copy of to support the inplace outputs. Wraps
 * embedding_bag_4bit_impl for templatization
 *
 * @param output Tensor where output should be written
 * @param weight Weight matrix for the embedding tables
 * @param indices Index for the embedding tables
 * @param offsets_in Offsets for the embedding tables
 * @param bit_width Number of bits used for storing embedding bag weights
 * @param num_embedding_bags The total number of embedding tables
 * @param output_size The output size of the embedding bags
 * @param idx Index of the current embedding table from the total number of
 * embedding tables
 */
void qembedding_bag_nbit_with_stride(
    at::Tensor& output,
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets_in,
    const int bit_width,
    const int num_embedding_bags,
    const int output_size,
    const int idx);

} // namespace kernels

} // namespace pace

#endif // EMBEDDINGBAG_KERNEL_H
