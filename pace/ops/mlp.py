# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************


from typing import Optional

import torch

from pace.ops.linear import Linear
from pace.ops.base import OperatorBase
from pace.ops.fused_linear import _get_fused_linear
from pace.ops.enum import FusedOperatorType, BackendType, DataType


class MergedMLP(OperatorBase):

    @property
    def operator_type(self):
        return FusedOperatorType.FUSEDMLPLINEAR

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: str = "relu",
        gate: bool = False,
        dtype: Optional[DataType] = None,
        backend_impl: BackendType = BackendType.NATIVE,
    ):
        super().__init__(backend_impl=backend_impl, dtype=dtype)

        self.in_features = in_features
        self.out_features = out_features
        self.bias_available = bias
        self.activation = activation
        self.gate = gate

        self.up_proj = _get_fused_linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            activation=activation if not gate else "mul",
            dtype=dtype,
            backend_impl=backend_impl if self.backend is None else None,
        )
        self.down_proj = Linear(
            in_features=out_features,
            out_features=in_features,
            bias=bias,
            dtype=dtype,
            backend_impl=backend_impl if self.backend is None else None,
        )
        if self.gate:
            self.gate_proj = _get_fused_linear(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
                activation=activation,
                dtype=dtype,
                backend_impl=backend_impl if self.backend is None else None,
            )
        else:
            self.gate_proj = None

        self._forward_impl = self._forward_impl_fallback
        if self.backend is not None:
            self._forward_impl = self._forward_impl_backend

    def _forward_impl_fallback(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate:
            out = self.gate_proj(x)
            out = self.up_proj(x, out)
        else:
            out = self.up_proj(x)
        return self.down_proj(out)

    def _forward_impl_backend(self, x: torch.Tensor) -> torch.Tensor:
        return self.backend.execute(
            x,
            self.up_proj_weight_chunks,
            self.up_proj_bias_chunks,
            self.down_proj_weight_chunks,
            self.down_proj.bias,
            self.activation,
            self.gate_proj_weight_chunks if self.gate else None,
            self.gate_proj_bias_chunks if self.gate else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias_available}, "
            f"activation={self.activation}, "
            f"gate={self.gate}, "
            f"dtype={self.dtype}, "
            f"backend_impl={self.backend if self.backend is not None else self.up_proj.backend},"
        )
