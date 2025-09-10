# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import torch

eps = torch.tensor(torch.finfo(torch.float32).eps)


def quantize_per_tensor(atensor: torch.Tensor, dtype: torch.dtype = torch.quint8):
    """
    Quantize a tensor to a given data type using per tensor quantization (with zero_point)
    Using Min-Max quantization

    Args:
    atensor: torch.Tensor
        The input tensor to be quantized
    dtype: torch.dtype
        The data type to which the tensor should be quantized

    Returns:
    torch.Tensor
        The quantized tensor
    """
    # Get the data type's minimum and maximum representable values
    qmin = torch.iinfo(dtype).min
    qmax = torch.iinfo(dtype).max

    # Calculate the minimum and maximum values of the tensor
    min_value = atensor.min().item()
    max_value = atensor.max().item()

    # Calculate the scale and zero_point
    scale = torch.tensor((max_value - min_value) / (qmax - qmin))
    scale = torch.max(scale, eps)

    zero_point = qmin - torch.round(min_value / scale).to(torch.int)
    zero_point = torch.clamp(zero_point, qmin, qmax)

    return torch.quantize_per_tensor(atensor, scale, zero_point, dtype)


def quantize_per_tensor_without_zeropoint(
    atensor: torch.Tensor, dtype: torch.dtype = torch.quint8
):
    """
    Quantize a tensor to a given data type using per tensor quantization (without zero_point)
    Using Abs-Max quantization (per tensor)

    Args:
    atensor: torch.Tensor
        The input tensor to be quantized
    dtype: torch.dtype
        The data type to which the tensor should be quantized

    Returns:
    torch.Tensor
        The quantized tensor
    """
    # Get the data type's minimum and maximum representable values
    qmin = torch.iinfo(dtype).min
    qmax = torch.iinfo(dtype).max

    # Calculate the minimum and maximum values of the tensor
    max_value = torch.max(torch.abs(atensor))

    # Calculate the scale and zero_point
    scale = max_value / (float(qmax - qmin) / 2)
    scale = torch.max(scale, eps)
    zero_point = torch.zeros_like(scale, dtype=torch.int64)

    return torch.quantize_per_tensor(atensor, scale, zero_point, dtype)


def quantize_per_channel(atensor: torch.Tensor, dtype: torch.dtype = torch.quint8):
    """
    Quantize a tensor to a given data type using per channel quantization (without zero_point)
    Using Abs-Max quantization (per channel)

    Args:
    atensor: torch.Tensor
        The input tensor to be quantized
    dtype: torch.dtype
        The data type to which the tensor should be quantized

    Returns:
    torch.Tensor
        The quantized tensor
    """

    # Get the data type's minimum and maximum representable values
    qmin = torch.iinfo(dtype).min
    qmax = torch.iinfo(dtype).max

    # Calculate the minimum values of the tensor along each channel
    max_value = torch.max(torch.abs(atensor), -1)[0]

    # Calculate the scale and zero_point
    scale = max_value / (float(qmax - qmin) / 2)
    scale = torch.max(scale, eps)
    zpoint = torch.zeros_like(scale, dtype=torch.int64)
    return torch.quantize_per_channel(atensor, scale, zpoint, 0, dtype)
