/*******************************************************************************
 * Modifications Copyright (c) 2024 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 * Portions of this file consist of AI-generated content
 *******************************************************************************/

#ifndef EMBEDDINGBAG_H
#define EMBEDDINGBAG_H

#include <ATen/ATen.h>

namespace pace {

/**
 * @brief The op registered to fuse quantized::embedding_bag_[byte/4bit]
 * Fusion happens at csrc/graph/fuse_embedding_bags.h
 *
 * @param weights weights listed as EmbeddingPackedParamsBase
 * @param indices Indexes to be used for embedding tables
 * @param offsets Offsets to be used for embedding tables
 * @param dense The dense input which is to be merged with the embedding bag
 * @param bit_width Number of bits used for storing embedding bag weights
 * outputs
 * @return at::Tensor
 */
at::Tensor qmerged_embedding_bag_nbit_cat(
    const c10::ArrayRef<c10::intrusive_ptr<EmbeddingPackedParamsBase>> weights,
    const at::TensorList& indices,
    const at::TensorList& offsets,
    const at::Tensor& dense,
    const int64_t bit_width);
} // namespace pace

#endif // EMBEDDINGBAG_H
