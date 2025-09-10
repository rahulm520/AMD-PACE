# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************
from typing import List, Tuple, Optional
from itertools import product

import torch
from torch import nn
from torch.testing._internal.common_utils import TestCase

from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import (
    QConfigMapping,
    float_qparams_weight_only_qconfig,
    float_qparams_weight_only_qconfig_4bit,
)

import pace


def prepare_embeddingbag_input(
    embedding_dim: Optional[int] = None,
    embeddings: Optional[int] = None,
    index_dtype: torch.dtype = torch.int64,
    offset_dtype: torch.dtype = torch.int64,
    dense_dtype: torch.dtype = torch.float,
):
    min_power = 1
    max_power = 10
    max_num_embedding_bag = 50

    # These will only be generated once for calibration, reused for testing
    if embedding_dim is None:
        embedding_dim = (
            2 ** torch.randint(low=min_power, high=max_power, size=(1,)).item()
        )
    if embeddings is None:
        num_embeddings = torch.randint(
            low=min_power, high=max_num_embedding_bag, size=(1,)
        ).item()
        embeddings = torch.randint(
            low=1, high=2**max_power, size=(num_embeddings,)
        ).tolist()
    else:
        num_embeddings = len(embeddings)

    # Prepare the input
    batch_size = 2 ** torch.randint(low=min_power, high=max_power, size=(1,)).item()
    multi_hot = torch.randint(low=1, high=10, size=(num_embeddings,)).tolist()
    lsi = [torch.ones((batch_size * h), dtype=index_dtype) for h in multi_hot]
    lso = [
        torch.arange(0, (batch_size + 1) * h, h, dtype=offset_dtype) for h in multi_hot
    ]
    dsx = torch.randn((batch_size, embedding_dim), dtype=dense_dtype)

    return embedding_dim, embeddings, (lsi, lso, dsx)


class MergedEmbeddingBagModel(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: List[int],
    ) -> None:

        super().__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = len(num_embeddings)
        self.embedding_bags: nn.ModuleList = nn.ModuleList()
        for num_embeddings in num_embeddings:
            EE = torch.nn.EmbeddingBag(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                include_last_offset=True,
                mode="sum",
            )
            self.embedding_bags.append(EE)

    def forward(
        self,
        index: List[torch.Tensor],
        offset: List[torch.Tensor],
        dense: torch.Tensor,
    ) -> torch.Tensor:
        B = offset[0].numel() - 1

        res = []  # removed list comprehension
        for idx in range(len(self.embedding_bags)):
            e, i, o = self.embedding_bags[idx], index[idx], offset[idx]
            res.append(e(i, o))
        res = [dense] + res

        data = torch.cat(res, dim=1).reshape(
            B, (self._num_embeddings + 1) * self._embedding_dim
        )
        return data


def quantize_embeddingbag_model(
    model: nn.Module,
    nbits: int,
    random_inputs: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor],
):

    if nbits == 4:
        emb_qconfig = float_qparams_weight_only_qconfig_4bit
    else:
        emb_qconfig = float_qparams_weight_only_qconfig
    static_qconfig_mapping = QConfigMapping().set_global(emb_qconfig)
    model = prepare_fx(model, static_qconfig_mapping, example_inputs=(*random_inputs,))
    model(*random_inputs)
    model = convert_fx(model)

    # Trace, Freeze and return the model
    model.eval()
    model = torch.jit.trace(model, (*random_inputs,), check_trace=True)
    model = torch.jit.freeze(model)
    return model


class TestMergedEmbeddingBag(TestCase):

    # This test should be equal at every precision since
    # the new op is an inplace write operator with no
    # numerical computation differences
    def test_merged_embeddingbag(self):

        for nbits in [4, 8]:

            embedding_dim, num_embeddings, calib_inputs = prepare_embeddingbag_input()
            model = MergedEmbeddingBagModel(
                embedding_dim=embedding_dim, num_embeddings=num_embeddings
            )
            qmodel = quantize_embeddingbag_model(
                model, nbits=nbits, random_inputs=calib_inputs
            )

            random_input_combo = []
            for index_dtype, offset_dtype in product(
                [torch.int64, torch.int32], [torch.int64, torch.int32]
            ):
                _, _, random_inputs = prepare_embeddingbag_input(
                    embedding_dim, num_embeddings, index_dtype, offset_dtype
                )
                random_input_combo.append(random_inputs)

            # Getting all the outputs from reference model now
            # since after conversion to pace model, the model
            # will be inplace
            ref_out = []
            for random_inputs in random_input_combo:
                ref_out.append(qmodel(*random_inputs))

            # Enable PACE Fusion to fuse the EmbeddingBag and concat
            # and run inputs through the model to enable it
            pace.core.enable_pace_fusion(True)
            qmodel(*random_inputs)
            qmodel(*random_inputs)

            pace_out = []
            for random_inputs in random_input_combo:
                pace_out.append(qmodel(*random_inputs))

            for reference_output, pace_output in zip(ref_out, pace_out):
                self.assertEqual(reference_output, pace_output)

    @classmethod
    def get_quantized_model(cls, nbits: int):
        embedding_dim, num_embeddings, calib_inputs = prepare_embeddingbag_input()
        model = MergedEmbeddingBagModel(
            embedding_dim=embedding_dim, num_embeddings=num_embeddings
        )
        qmodel = quantize_embeddingbag_model(
            model, nbits=nbits, random_inputs=calib_inputs
        )

        # Enable PACE Fusion to fuse the EmbeddingBag
        pace.core.enable_pace_fusion(True)
        qmodel(*calib_inputs)
        qmodel(*calib_inputs)

        return qmodel, embedding_dim, num_embeddings

    # Most of the invalid dtypes will be handled by the standard
    # implementation since the fusion happens online and the model
    # cannot be serialized after that. This test is to check if the
    # model is able to handle invalid inputs
    def test_merged_embeddingbag_invalid_input(self):
        qmodel, embedding_dim, num_embeddings = (
            TestMergedEmbeddingBag.get_quantized_model(4)
        )

        _, _, random_inputs = prepare_embeddingbag_input(embedding_dim, num_embeddings)
        lsi, lso, dsx = random_inputs

        # Invalid input
        with self.assertRaisesRegex(RuntimeError, "missing value"):
            qmodel(lsi, lso)

        # 1st argument is self, so 4 arguments are expected
        with self.assertRaisesRegex(RuntimeError, "expected at most 4 argument"):
            qmodel(lsi, lso, dsx, dsx)

        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected a value of type 'List\[Tensor\]' for argument 'index'",
        ):
            qmodel(lsi[0], lso, dsx)

        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected a value of type 'List\[Tensor\]' for argument 'offset'",
        ):
            qmodel(lsi, lso[0], dsx)

        with self.assertRaisesRegex(
            RuntimeError, "Expected a value of type 'Tensor' for argument 'dense'"
        ):
            qmodel(lsi, lso, lsi)

    def test_merged_embeddingbag_invalid_dtypes(self):
        qmodel, embedding_dim, num_embeddings = (
            TestMergedEmbeddingBag.get_quantized_model(4)
        )

        _, _, random_inputs = prepare_embeddingbag_input(
            embedding_dim, num_embeddings, index_dtype=torch.float32
        )
        with self.assertRaisesRegex(
            RuntimeError, "Expected Int or Long indices, but found"
        ):
            qmodel(*random_inputs)

        _, _, random_inputs = prepare_embeddingbag_input(
            embedding_dim, num_embeddings, offset_dtype=torch.float32
        )
        with self.assertRaisesRegex(
            RuntimeError, "Expected Int or Long offsets, but found"
        ):
            qmodel(*random_inputs)

        _, _, random_inputs = prepare_embeddingbag_input(
            embedding_dim, num_embeddings, dense_dtype=torch.bfloat16
        )
        with self.assertRaisesRegex(
            RuntimeError, "Expected Float dense input, but found"
        ):
            qmodel(*random_inputs)

    def test_merged_embeddingbag_non_contiguous(self):
        qmodel, embedding_dim, num_embeddings = (
            TestMergedEmbeddingBag.get_quantized_model(4)
        )
        _, _, (lsi, lso, dsx) = prepare_embeddingbag_input(
            embedding_dim, num_embeddings
        )

        with self.assertRaisesRegex(
            RuntimeError, "Expected weight, indices, and offsets to be contiguous"
        ):
            # Make the indices non-contiguous by cloning the tensor and taking every other element
            non_contiguous_lsi = [torch.cat([lsi_v, lsi_v])[::2] for lsi_v in lsi]
            qmodel(non_contiguous_lsi, lso, dsx)

        with self.assertRaisesRegex(
            RuntimeError, "Expected weight, indices, and offsets to be contiguous"
        ):

            # To make a non-contiguous offset, we can clone the tensor and take every other element
            # and then set the last element to the original last element (since it has to be the same)
            # as the number of indices
            non_contiguous_lso = []
            for lso_val in lso:
                non_contiguous_lso_val = torch.cat([lso_val, lso_val])[::2]
                non_contiguous_lso_val[-1] = lso_val[-1]
                non_contiguous_lso.append(non_contiguous_lso_val)

            qmodel(lsi, non_contiguous_lso, dsx)

        with self.assertRaisesRegex(
            RuntimeError, "Expected dense input to be contiguous"
        ):
            non_contiguous_dsx = dsx.transpose(0, 1)
            qmodel(lsi, lso, non_contiguous_dsx)

    def test_merged_embeddingbag_invalid_shape(self):
        qmodel, embedding_dim, num_embeddings = (
            TestMergedEmbeddingBag.get_quantized_model(4)
        )

        _, _, (lsi, lso, dsx) = prepare_embeddingbag_input(
            embedding_dim, num_embeddings
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Number of elements in offsets should be equal to the number of indices",
        ):
            qmodel(lsi[:-1], lso, dsx)

        # The torchscript interpreter itself will throw an error if the number of elements
        # in the list is not equal to the number of embeddings
        with self.assertRaisesRegex(
            RuntimeError, "Expected [0-9]+ elements in a list but found [0-9]+"
        ):
            qmodel(lsi, lso[:-1], dsx)

        with self.assertRaisesRegex(
            RuntimeError, "Expected output size is [0-9]+ but recieved [0-9]+ instead"
        ):
            qmodel(lsi, lso, torch.rand(dsx.shape[0], dsx.shape[1] + 1))

        _, _, new_random_inputs = prepare_embeddingbag_input(
            embedding_dim - 1, num_embeddings
        )
        with self.assertRaisesRegex(
            RuntimeError, "Expected output size is [0-9]+ but recieved [0-9]+ instead*"
        ):
            qmodel(*new_random_inputs)

        _, _, new_random_inputs = prepare_embeddingbag_input()
        with self.assertRaisesRegex(
            RuntimeError, "Expected [0-9]+ elements in a list but found [0-9]+"
        ):
            qmodel(*new_random_inputs)
