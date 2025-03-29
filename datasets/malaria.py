import os
import json

import scipy
import numpy as np
import pandas as pd

import torch
import tqdm

from .base import BaseDataset

import time

class MalariaData:
    
    def __init__(self, path: str):
        """
        Malaria dataset, preprocessed (imputed) by Zhang et al. iScience (2022)

        Args:
            path (str): Path to the dataset. The path `path/SubCh2_TrainingData_imputed.csv`
            should exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError
        
        self.path = os.path.abspath(path)
        self.data_path = os.path.join(self.path, "SubCh2_TrainingData_imputed.csv")

        self.raw_data = pd.read_csv(self.data_path)

        data_cols = [colname for colname in self.raw_data.columns if colname.startswith("PF3D7_")]
        metadata_cols = [colname for colname in self.raw_data.columns if colname not in data_cols]

        self.data = self.raw_data.loc[:, data_cols]
        self.obs = self.raw_data.loc[:, metadata_cols]

        assert self.obs.index.is_unique

        return

    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, key):
        
        if not isinstance(key, (int, np.int8, np.int16, np.int32, np.int64, torch.long)):
            raise ValueError(f"Dataset cannot be indexed with an index of type {type(key)}")

        if key < 0:
            key += len(self)

        if not (0 <= key < len(self)):
            raise ValueError(f"Index out of bounds")
        
        x = self.data.iloc[key, :].to_numpy()
        metadata = self.obs.iloc[key, :]

        return (x, metadata)

class Malaria(BaseDataset):
    def __init__(self, 
            path: str=None,
            data: MalariaData=None
        ):
        if sum([path is None, data is None]) != 1:
            raise ValueError(f"Either data_path or data must be specified, but not both")
        
        self.path = path if path is not None else data.data_path

        self.data = MalariaData(path=path) if path is not None else data

        self._index = np.arange(len(self.data.obs))
    
    @property
    def obs(self):
        return self.data.obs.iloc[self._index, :]
    
    def subset(self, mask: np.ndarray=None, inplace=True):

        if mask is not None and not isinstance(mask, np.ndarray):
            raise ValueError(f"Mask must be a numpy array")
        if mask is not None and len(mask) != len(self._index):
            raise ValueError(f"Mask must specify one value for each observation")

        if mask is not None and mask.dtype not in (bool, np.bool_):
            print(f"WARNING: mask dtype is {mask.dtype} but bool is expected. We will cast the mask to bool.")
            mask = mask.astype(bool)

        if inplace:
            self._index = np.arange(len(self.data.obs)) if mask is None else self._index[mask]
            return
        
        dataset = Malaria(path=None, data=self.data)
        dataset._index = np.arange(len(self.data.obs)) if mask is None else self._index[mask]
    
        return dataset

    def __len__(self):
        return len(self._index)
    
    def __getitem__(self, key):
                
        if not isinstance(key, (int, np.int8, np.int16, np.int32, np.int64, torch.long)):
            raise ValueError(f"Dataset cannot be indexed with an index of type {type(key)}")

        if key < 0:
            key += len(self)

        if not (0 <= key < len(self)):
            raise ValueError(f"Index out of bounds")

        return self.data[self._index[key]]

