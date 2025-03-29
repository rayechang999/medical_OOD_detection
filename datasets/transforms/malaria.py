import os
import json

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

class RankNormalization:
    def __init__(self, data: np.ndarray):
        """
        Parameters
        ----------
        data : np.ndarray
            The full dataset. Required for rank normalization.
        """
        self.ranked = np.mean(np.sort(data, axis=1), axis=0)

    def __call__(self, x: np.ndarray):
        x_rank = np.argsort(np.argsort(x))
        x = self.ranked[x_rank]

        # first argsort: move the i-th index to the position where it would need to be,
        # relative to all the other indices, for the array to be sorted (i.e. move the index)
        # to its rank *position*
        # second argsort: get that rank position as an integer, so the position of index i
        # is now the value at index i
        
        return x

class RandomMagnitudeRescaling:
    def __init__(self, scale_factor_min=0.8, scale_factor_max=1.2):
        self.scale_factor_min = scale_factor_min
        self.scale_factor_max = scale_factor_max
    
    def __call__(self, x):
        r = np.random.uniform(low=self.scale_factor_min, high=self.scale_factor_max)

        return r * x