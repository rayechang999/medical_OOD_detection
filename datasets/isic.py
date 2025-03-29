import os
import json

import scipy
import numpy as np
import pandas as pd

import soundfile

import torch

import PIL

import tqdm

from .base import BaseDataset

import time

class ISICData:
    
    def __init__(self, data_path: str, metadata_path: str, verbose: bool=True):
        """
        ISIC Training 2020 dataset. To use this dataset, download and unpack locally. The 
        path argument of this class should be set to the unpacked folder.

        Args:
            path (str): Path to the ISIC dataset, the extracted ZIP file.
            verbose (bool): Whether or not to print 
        
        Examples:
            Loading in the ISIC dataset:
                ```
                from datasets import ISIC

                data = ISIC("../download", "../challenge-2020-training_metadata_2024-11-22.csv")
                ```
        """
        self.data_path = os.path.abspath(data_path)
        self.metadata_path = os.path.abspath(metadata_path)

        self.metadata = pd.read_csv(metadata_path)
        # Check that metadata_path metadata is the same as the data_path metadata
        assert pd.read_csv(os.path.join(data_path, "metadata.csv")).equals(self.metadata)

        # Each row of metadata contains a particular recording observation
        index = list()
        df = {column: list() for column in self.metadata.columns}
        df["file::path"] = list()

        for file_name in tqdm.tqdm(list(sorted(os.listdir(data_path)))):
            
            if not file_name.endswith(".jpg"):
                print(f"Skipping file {file_name}", time.perf_counter())
                print()
                continue
            
            row = self.metadata.loc[self.metadata["isic_id"] == os.path.splitext(file_name)[0], :]
            assert len(row) == 1, (os.path.splitext(file_name)[0], file_name, row)

            for column in self.metadata.columns:
                df[column].append(row[column].iloc[0])
            
            index.append(os.path.splitext(file_name)[0])
            df["file::path"].append(os.path.join(data_path, file_name))
        
        assert np.all(np.unique(index, return_counts=True)[1] == 1)
        
        self.obs = pd.DataFrame(df, index=index)
        self.obs["data::width"] = float("nan")
        self.obs["data::height"] = float("nan")
        self.obs["data::mode"] = ""

        self.data = None

    def _fetch_file(self, index) -> tuple:
        """
        Fetches a single WAV file, returns the sampling rate and data.
        """
        path = self.obs.loc[self.obs.index[index], "file::path"]

        with PIL.Image.open(path) as im:
            image = im.copy()

        self.obs.loc[self.obs.index[index], "data::width"] = image.size[0]
        self.obs.loc[self.obs.index[index], "data::height"] = image.size[1]
        self.obs.loc[self.obs.index[index], "data::mode"] = image.mode

        return image

    def fetch(self):
        """
        Fetches the full dataset from disk into memory.
        """
        fetched_data = list()
        for path in self.obs["coswara_file::path"]:
            data = self._fetch_file(path)
            fetched_data.append(data)
        
        self.data = np.array(fetched_data, dtype=object) # keep as list
        self.obs["data::width"] = [image.size[0] for image in self.data]
        self.obs["data::height"] = [image.size[1] for image in self.data]
        self.obs["data::mode"] = [image.mode for image in self.data]

    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, key):
        
        if not isinstance(key, (int, np.int8, np.int16, np.int32, np.int64, torch.long)):
            raise ValueError(f"Dataset cannot be indexed with an index of type {type(key)}")

        if key < 0:
            key += len(self)

        if not (0 <= key < len(self)):
            raise ValueError(f"Index out of bounds")
        
        item = self.data[key] if self.data is not None else self._fetch_file(key)
        metadata = self.obs.iloc[key, :]

        x = (item, metadata)

        return x

class ISIC(BaseDataset):
    def __init__(self, 
            data_path: str=None, 
            metadata_path: str=None,
            verbose: bool=True, 
            data: ISICData=None
        ):
        if sum([data_path is None, data is None]) != 1:
            raise ValueError(f"Either data_path or data must be specified, but not both")
        
        if data_path is not None and metadata_path is None:
            raise ValueError(f"If data_path is specified, metadata_path must also be specified")
        
        self.data_path = data_path if data_path is not None else data.data_path
        self.metadata_path = metadata_path if data_path is not None else data.metadata_path

        self.data = ISICData(data_path=data_path, metadata_path=metadata_path, verbose=verbose) if data_path is not None else data

        self._index = np.arange(len(self.data.obs))
    
    @property
    def obs(self):
        return self.data.obs.iloc[self._index, :]
    
    def subset(self, mask: np.ndarray=None, inplace=True):
        # print("In subset") # DEBUG
        if mask is not None and not isinstance(mask, np.ndarray):
            raise ValueError(f"Mask must be a Boolean numpy array")
        if mask is not None and len(mask) != len(self._index):
            raise ValueError(f"Mask must specify one value for each observation")

        if mask is not None and mask.dtype not in (bool, np.bool):
            print(f"WARNING: mask dtype is {mask.dtype} but np.bool is expected. We will cast the mask to np.bool.")
            mask = mask.astype(bool)

        if inplace:
            self._index = np.arange(len(self.data.obs)) if mask is None else self._index[mask]
            # print(f"After subsetting, new index: {self._index}") # DEBUG
            return
        
        isic = ISIC(data_path=None, metadata_path=None, verbose=False, data=self.data)
        isic._index = np.arange(len(isic.data.obs)) if mask is None else self._index[mask]

        # print(f"After subsetting, new index: {isic._index}") # DEBUG
    
        return isic
    
    def fetch(self):
        return self.data.fetch()

    @property
    def fetched(self):
        return self.data.fetched
    
    def fetch_obs(self):
        if self.data.fetched:
            return
        
        for index in tqdm.tqdm(range(len(isic.data.obs)), desc="Fetching obs"):
            img = self.data[index] # automatically updates the obs for that image
            del img
        
        return

    def __len__(self):
        return len(self._index)
    
    def __getitem__(self, key):
                
        if not isinstance(key, (int, np.int8, np.int16, np.int32, np.int64, torch.long)):
            raise ValueError(f"Dataset cannot be indexed with an index of type {type(key)}")

        if key < 0:
            key += len(self)

        if not (0 <= key < len(self)):
            raise ValueError(f"Index out of bounds")
        
        # print(f"Key indexed: {key}") # DEBUG

        return self.data[self._index[key]]

