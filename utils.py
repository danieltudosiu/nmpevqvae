import torch


def min_max_scale(input_tensor):
    input_tensor_min, _ = input_tensor.view(input_tensor.shape[0], -1).min(1)
    input_tensor_min = input_tensor_min[
        (...,) + (None,) * (len(input_tensor.shape) - len(input_tensor_min.shape))
    ]

    input_tensor = input_tensor - input_tensor_min

    input_tensor_max, _ = input_tensor.view(input_tensor.shape[0], -1).max(1)
    input_tensor_max = input_tensor_max[
        (...,) + (None,) * (len(input_tensor.shape) - len(input_tensor_max.shape))
    ]

    input_tensor = input_tensor / input_tensor_max

    return input_tensor
