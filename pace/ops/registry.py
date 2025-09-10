# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from collections import defaultdict
from typing import Type, Dict, Tuple, List, Union

from pace.utils.logging import PACE_ASSERT, PACE_DEBUG
from pace.ops.enum import (
    OperatorType,
    FusedOperatorType,
    BackendType,
    DataType,
    FALLBACK_BACKEND,
)


class BackendRegistry:
    def __init__(self):
        self._registry: Dict[OperatorType, Dict[Tuple[BackendType, DataType], Type]] = (
            defaultdict(dict)
        )

    def register(
        self,
        operator: Union[OperatorType, FusedOperatorType],
        backend: BackendType,
        dtype: List[DataType],
    ):
        # Throw an error for duplicate registration of the combination of operator, backend and dtype
        if operator in self._registry and backend in self._registry[operator]:
            for dt in dtype:
                if (backend, dt) in self._registry[operator]:
                    PACE_ASSERT(
                        False,
                        f"Duplicate registration of {operator} with backend {backend} and dtype {dt}",
                    )

        def decorator(cls):
            for dt in dtype:
                key = (backend, dt)
                self._registry[operator][key] = cls
            return cls

        return decorator

    def get_available_backends(
        self, operator: Union[OperatorType, FusedOperatorType]
    ) -> List[Tuple[BackendType, DataType]]:
        PACE_DEBUG(f"Getting available backends for {operator}")

        if operator not in self._registry:
            PACE_DEBUG(f"Operator {operator} not registered, returning empty list")
            return []

        op_backends = self._registry[operator]
        return list(op_backends.keys())

    def get(
        self,
        operator: Union[OperatorType, FusedOperatorType],
        backend: BackendType,
        dtype: DataType,
    ):
        PACE_DEBUG(f"Getting backend for {operator}, {backend}, {dtype}")

        # if operator is not registered, return None
        if operator not in self._registry:
            PACE_DEBUG(
                f"Operator {operator} not registered, this operator might be a new one, or one without a backend"
            )
            return None

        op_backends = self._registry.get(operator, {})
        key = (backend, dtype)
        if key in op_backends:
            PACE_DEBUG(f"Found backend {backend} for {operator} with {dtype}")
            return op_backends[key]

        if isinstance(operator, FusedOperatorType):
            # Fused operators are not registered in the registry
            PACE_DEBUG(
                f"Fused operator {operator} with backend {backend} and dtype {dtype} not found in registry."
                "It will be handled by the default forward method."
            )
            return None

        PACE_DEBUG(
            f"Backend {backend} not found for {operator} with {dtype}. Trying fallback backends..."
        )
        for fallback_backend in FALLBACK_BACKEND:  # Default backend fallback
            key = (fallback_backend, dtype)
            if key in op_backends:
                PACE_DEBUG(
                    f"Using fallback backend {fallback_backend} for {operator} with {dtype}"
                )
                return op_backends[key]
        PACE_ASSERT(False, f"No backend found for {operator} with {backend}, {dtype}")

    def __repr__(self):
        return str(self._registry)


backend_registry = BackendRegistry()
