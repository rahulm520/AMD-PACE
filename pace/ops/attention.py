# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import Optional

import torch

from pace.ops.base import OperatorBase
from pace.ops.enum import OperatorType, BackendType, DataType


class MultiHeadAttention(OperatorBase):

    @property
    def operator_type(self):
        return OperatorType.MHA

    def __init__(
        self,
        dtype: Optional[DataType] = None,
        backend_impl: BackendType = BackendType.NATIVE,
    ):
        self.dtype = dtype
        super().__init__(backend_impl=backend_impl, dtype=dtype)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.backend.execute(
            query=query, key=key, value=value, attention_mask=attention_mask
        )

    def extra_repr(self):
        return f"dtype={self.dtype}, backend={self.backend}"
