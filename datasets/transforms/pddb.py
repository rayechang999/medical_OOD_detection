import os
import json

import scipy
import numpy as np
import pandas as pd

import torch
import torchvision

class ApplyTo:
    def __init__(self, transform, index: int):
        """
        Applies the given transform to the index-th element of an input tensor.
        For example, t = ApplyTo(Normalize, 0); t([x, y]) will normalize x, but 
        not y, and return both.
        """
        self.transform = transform
        self.index = index

    def __call__(self, args):
        args = list(args)
        args[self.index] = self.transform(args[self.index])
        return args

class Transpose:
    def __init__(self):
        pass
    
    def __call__(self, x):
        return x.T

class AxisNormalization:
    def __init__(self, axis):
        """
        Normalizes the input along a given axis.
        """
        self.axis = axis
    
    def __call__(self, x):
        x = (x - np.mean(x, axis=self.axis, keepdims=True)) / np.std(x, axis=self.axis, keepdims=True)

        return x

class RandomRotation:
    def __init__(self):
        pass
    
    def __call__(self, x):
        rot = scipy.spatial.transform.Rotation.random().as_matrix()

        x = rot @ x
        return x
    
class TemporalRescaling:
    def __init__(self, scale_factor_min=0.8, scale_factor_max=1.2):
        self.scale_factor_min = scale_factor_min
        self.scale_factor_max = scale_factor_max
    
    def __call__(self, x):
        assert np.shape(x)[0] == 3

        x_tensor = torch.tensor(x).unsqueeze(1) # shape (3, 1, T), so basically treating like an image

        T = np.shape(x)[1]
        new_T = int(np.shape(x)[1] * np.random.uniform(low=self.scale_factor_min, high=self.scale_factor_max))
        x_tensor = torchvision.transforms.Resize((1, new_T))(x_tensor)
        
        x = x_tensor.squeeze(1).numpy()

        if new_T > T:
            # Crop: take a central segment of size T
            start_idx = (new_T - T) // 2
            perturbed_data = x[:, start_idx:start_idx + T]
        else:
            # Pad: add zeros or replicate edges to match size T
            pad_width = (T - new_T) // 2
            perturbed_data = np.pad(
                x, 
                pad_width=((0, 0), (pad_width, T - new_T - pad_width)), 
                mode="constant"
            )

        return x
    
class MagnitudeRescaling:
    def __init__(self, scale_factor_min=0.8, scale_factor_max=1.2):
        self.scale_factor_min = scale_factor_min
        self.scale_factor_max = scale_factor_max
    
    def __call__(self, x):
        r = np.random.uniform(low=self.scale_factor_min, high=self.scale_factor_max)

        return r * x

class SetSize:
    def __init__(self, dim: int, size: int):
        self.dim = dim
        self.size = size

    def __call__(self, x: np.ndarray):
        current_size = np.shape(x)[self.dim]

        if current_size < self.size:
            pad_width = [(0, 0)] * x.ndim
            pad_width[self.dim] = (0, self.size - current_size)
            x = np.pad(x, pad_width, mode='constant')

        elif current_size > self.size:
            slices = [slice(None)] * x.ndim
            slices[self.dim] = slice(0, self.size)
            x = x[tuple(slices)]

        return x

class ConvertYToTensor:
    def __init__(self, metadata_to_y_func, y_dtype):
        """
        Applies a transform to only a given element within a tuple. This is 
        to be used as a wrapper and is useful for dataset loading classes which 
        return a tuple (x, metadata) instead of just x.
        """
        self.metadata_to_y_func = metadata_to_y_func
        self.y_dtype = y_dtype


    def __call__(self, metadata):

        y = torch.tensor(self.metadata_to_y_func(metadata), dtype=self.y_dtype)

        return y
    
class AsTensor:
    def __init__(self, dtype=torch.get_default_dtype(), device=torch.get_default_device()):
        self.dtype = dtype
        self.device = device

    def __call__(self, x):
        return torch.tensor(x, dtype=self.dtype, device=self.device)
    