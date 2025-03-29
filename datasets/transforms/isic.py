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

class Rescale: # not necessary if we're using pillow + transforms.ToTensor
    def __init__(self, scale_factor=255):
        self.scale_factor = scale_factor
    
    def __call__(self, x):
        x = x / self.scale_factor

        return x

class AddMetadata:
    def __init__(self, metadata_funcs: dict):
        """
        Maps columns of metadata to vectors and concatenates them together. Each 
        (key, value) pair in metadata_funcs should map strings (the name of the 
        column in the metadata) to a vector, which will be concatenated together 
        along the last axis.
        """
        self.metadata_funcs = metadata_funcs
    
    def __call__(self, x):
        x, metadata = x
        
        tensors = [self.metadata_funcs[key](metadata[key]) for key in self.metadata_funcs.keys()]
        # print([t.shape for t in tensors])
        x_metadata = torch.cat(tensors, dim=-1)
        
        return (x, x_metadata), metadata


class ConvertYToTensor:
    def __init__(self, metadata_to_y_func, y_dtype):
        """
        Applies a transform to only a given element within a tuple. This is 
        to be used as a wrapper and is useful for dataset loading classes which 
        return a tuple (x, metadata) instead of just x.
        """
        self.metadata_to_y_func = metadata_to_y_func
        self.y_dtype = y_dtype


    def __call__(self, x):
        x, metadata = x

        y = torch.tensor(self.metadata_to_y_func(metadata), dtype=self.y_dtype)

        return x, y

class FlattenX:
    def __init__(self):
        pass
    
    def __call__(self, x):
        x, y = x
        return x.view(-1), y

class Select:
    def __init__(self, index: int):
        self.index = index

    def __call__(self, inputs):
        return inputs[self.index]