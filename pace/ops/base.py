# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import Type, Optional
from abc import ABC, abstractmethod

from torch import nn
from torch import get_default_dtype

from pace.ops.registry import backend_registry
from pace.ops.enum import OperatorType, BackendType, DataType


class OperatorBase(nn.Module, ABC):
    """
    Base class for all operators in the PACE framework.

    This class provides a common interface for all operators, including
    initialization, parameter creation, weight loading, and forward pass.
    """

    def __init__(
        self,
        backend_impl: Optional[BackendType] = None,
        dtype: Optional[DataType] = None,
    ):

        super().__init__()
        dtype = dtype if dtype is not None else get_default_dtype()  # Get default dtype
        dtype = (
            dtype if isinstance(dtype, DataType) else DataType.from_torch(dtype)
        )  # Convert to DataType

        self.dtype = dtype
        self.backend_impl = backend_impl
        self.backend_cls: Type[BackendBase] = backend_registry.get(
            self.operator_type, backend_impl, dtype
        )
        if self.backend_cls is not None:
            self.backend: BackendBase = self.backend_cls()
            self.backend.create_extra_params(self)
        else:
            # There could be some operators that do not have a backend
            self.backend = None

    @property
    @abstractmethod
    def operator_type(cls) -> OperatorType:
        """
        Returns the type of the operator. Should be implemented by subclasses.
        """
        pass

    def load_weights(self, *args, **kwargs):
        """
        Weights loading method for the operator parameter.
        This method has to be called after initialization by the user manually.
        """
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward method for the operator. Should be implemented by subclasses.
        Will be called during the forward pass of the operator.
        """
        pass

    @abstractmethod
    def extra_repr(self) -> str:
        """
        Returns a string representation of the operator's parameters.
        Should be implemented by subclasses.
        """
        pass


class BackendBase(ABC):
    """
    Base class for backend implementations of operators.
    This class provides a common interface for backend-specific operations.
    """

    def create_extra_params(self, layer: nn.Module):
        """
        Create parameters for the backend. Can be implemented by subclasses.
        This method will be called during the operator's initialization.
        """
        pass

    def preprocess(self, layer: nn.Module):
        """
        Preprocess method for the backend. Optional for subclasses to implement.
        """
        pass

    @abstractmethod
    def execute(self, *args, **kwargs):
        """
        Execute method for the backend. Should be implemented by subclasses.
        The operator's forward pass will call this method, and the actual
        computation will be performed here.
        """
        pass

    def __repr__(self):
        """
        Returns a string representation of the backend.
        """
        return f"{self.__class__.__name__}"
