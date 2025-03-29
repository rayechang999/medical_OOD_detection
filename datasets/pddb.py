import os
import json

import scipy
import numpy as np
import pandas as pd

import torch
import tqdm

from .base import BaseDataset

import time

class PDDBData:
    
    def __init__(self, path: str, verbose: bool=True):
        """
        PDDB dataset, preprocessed by Zhang et al. Patterns (2020)

        Args:
            path (str): Path to Parkinson-new folder. The paths
                Parkinson-new/PD_walking_project/PDWalk should exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError
        
        self.path = os.path.abspath(path)
        self.data_path = os.path.join(self.path, "PD_walking_project", "PDwalk")

        demographic_survey = pd.read_csv(os.path.join(self.path, "Demographic_survey.tsv"), delimiter="\t")

        walking_activity_training = pd.read_csv(os.path.join(self.path, "PD_walking_project", "PDwalk", "Walking_Activity_training.csv"))
        str_cols = [col for col in walking_activity_training.columns if col.endswith(".json.items")]
        walking_activity_training = pd.read_csv(os.path.join(self.path, "PD_walking_project", "PDwalk", "Walking_Activity_training.csv"), dtype={col: str for col in str_cols})

        outbound_files = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(self.data_path, "outbound_50Hz")) if file.endswith(".npy")]
        rest_files = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(self.data_path, "rest_50Hz")) if file.endswith(".npy")]

        outbound_files = list(sorted(outbound_files))
        rest_files = list(sorted(rest_files))

        # Only keep walking_activity_training records for which we have a .npy array
        outbound_col = "accel_walking_outbound.json.items"
        rest_col = "accel_walking_rest.json.items"

        has_outbound_record = walking_activity_training[outbound_col].isin(outbound_files)
        has_rest_record = walking_activity_training[outbound_col].isin(rest_files)

        walking_activity_training = walking_activity_training.loc[has_outbound_record | has_rest_record, :]
        demographic_survey = demographic_survey.loc[demographic_survey["healthCode"].isin(walking_activity_training["healthCode"]), :]

        # Each row of metadata contains a particular recording observation
        index = list()
        df = {column: list() for column in walking_activity_training.columns.tolist() + demographic_survey.columns.tolist()}
        df["data::type"] = list()
        df["file::path"] = list()

        for file in tqdm.tqdm(outbound_files, desc="Load outbound"):

            row_wat = walking_activity_training.loc[walking_activity_training[outbound_col] == file, :]
            if len(row_wat) < 1:
                continue
            assert len(row_wat) == 1
            row_wat = row_wat.iloc[0, :]

            healthCode = row_wat["healthCode"]

            row_ds = demographic_survey.loc[demographic_survey["healthCode"] == healthCode, :]
            if len(row_ds) < 1:
                continue
            assert len(row_ds) == 1
            row_ds = row_ds.iloc[0, :]

            index.append(file)
            for column in df.keys():
                if column == "file::path":
                    df[column].append(os.path.join(self.data_path, "outbound_50Hz", f"{file}.npy"))
                elif column == "data::type":
                    df[column].append("outbound_50Hz")
                elif column in walking_activity_training.columns:
                    df[column].append(row_wat[column])
                elif column in demographic_survey.columns:
                    df[column].append(row_ds[column])
                else:
                    raise Exception
        
        for file in tqdm.tqdm(rest_files, desc="Load rest"):

            row_wat = walking_activity_training.loc[walking_activity_training[rest_col] == file, :]
            if len(row_wat) < 1:
                continue
            assert len(row_wat) == 1
            row_wat = row_wat.iloc[0, :]

            healthCode = row_wat["healthCode"]
            
            row_ds = demographic_survey.loc[demographic_survey["healthCode"] == healthCode, :]
            if len(row_ds) < 1:
                continue
            assert len(row_ds) == 1
            row_ds = row_ds.iloc[0, :]

            index.append(file)
            for column in df.keys():
                if column == "file::path":
                    df[column].append(os.path.join(self.data_path, "rest_50Hz", f"{file}.npy"))
                elif column == "data::type":
                    df[column].append("rest_50Hz")
                elif column in walking_activity_training.columns:
                    df[column].append(row_wat[column])
                elif column in demographic_survey.columns:
                    df[column].append(row_ds[column])
                else:
                    raise Exception

        self.walking_activity_training = walking_activity_training
        self.demographic_survey = demographic_survey
        
        self.obs = pd.DataFrame(df, index=index)
        assert self.obs.index.is_unique

        # self.obs["professional-diagnosis"] = self.obs["professional-diagnosis"].astype(float)

        return

    def _fetch_file(self, index) -> tuple:
        """
        Fetches a single time series; returns only this time series.
        """
        path = self.obs.loc[self.obs.index[index], "file::path"]

        arr = np.load(path)

        return arr

    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, key):
        
        if not isinstance(key, (int, np.int8, np.int16, np.int32, np.int64, torch.long)):
            raise ValueError(f"Dataset cannot be indexed with an index of type {type(key)}")

        if key < 0:
            key += len(self)

        if not (0 <= key < len(self)):
            raise ValueError(f"Index out of bounds")
        
        item = self._fetch_file(key)
        metadata = self.obs.iloc[key, :]

        x = (item, metadata)

        return x

class PDDB(BaseDataset):
    def __init__(self, 
            path: str=None, 
            verbose: bool=True, 
            data: PDDBData=None
        ):
        if sum([path is None, data is None]) != 1:
            raise ValueError(f"Either data_path or data must be specified, but not both")
        
        self.path = path if path is not None else data.data_path

        self.data = PDDBData(path=path, verbose=verbose) if path is not None else data

        self._index = np.arange(len(self.data.obs))
    
    @property
    def obs(self):
        return self.data.obs.iloc[self._index, :]
    
    def subset(self, mask: np.ndarray=None, inplace=True):
        # print("In subset") # DEBUG
        if mask is not None and not isinstance(mask, np.ndarray):
            raise ValueError(f"Mask must be a numpy array")
        if mask is not None and len(mask) != len(self._index):
            raise ValueError(f"Mask must specify one value for each observation")

        if mask is not None and mask.dtype not in (bool, np.bool_):
            print(f"WARNING: mask dtype is {mask.dtype} but np.bool is expected. We will cast the mask to np.bool.")
            mask = mask.astype(bool)

        if inplace:
            self._index = np.arange(len(self.data.obs)) if mask is None else self._index[mask]
            # print(f"After subsetting, new index: {self._index}") # DEBUG
            return
        
        pddb = PDDB(path=None, data=self.data)
        pddb._index = np.arange(len(isic.data.obs)) if mask is None else self._index[mask]

        # print(f"After subsetting, new index: {isic._index}") # DEBUG
    
        return pddb

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

