import os
import json

import scipy
import numpy as np
import pandas as pd

import torch

import tqdm

from .base import BaseDataset

class GaussianGenerator:
    
    def __init__(self, config: dict, seed: int=None):
        """
        Parameters
        ----------
        config : dict
            A dictionary, where each key is the name of a different 
            Gaussian to sample from. Each value contains the settings for that 
            Gaussian, including keys "dim": int (dimensionality), 
            "num_samples": int (number of samples), and optional
            keys for "mean": np.array (mean) and "cov": np.array (covariance matrix).
            (If unspecified, these means/covariances are randomly generated.)
            Example config:
            {
                "gaussian_0": {
                    "dim": 2,
                    "num_samples": 500,
                    "mean": np.array([0.1, -0.9])
                    # Here, the covariance matrix will be randomly generated
                },
                "guassian_1": {
                    "dim": 4,
                    "num_samples": 500,
                    # Here, the mean will be randomly generated
                    "cov": np.identity(4), # ensure this is symmetric and positive semi-definite
                }
            }
        seed : int
            A seed for reproducibility.
        """
        self.original_config = config
        self.seed = seed

        self.rng = np.random.default_rng(self.seed)

        # Generate mean and cov
        self.config = dict()
        for name in config.keys():
            mean = config[name]["mean"] if "mean" in config[name].keys() else self.rng.normal(size=config[name]["dim"])
            if "cov" in config[name].keys():
                cov = config[name]["cov"]
            else:
                A = self.rng.normal(size=(config[name]["dim"], config[name]["dim"]))
                cov = A @ A.T
            
            self.config[name] = {"mean": mean, "cov": cov, "dim": config[name]["dim"], "num_samples": config[name]["num_samples"]}

        # Generate samples
        data = list()
        obs = {"name": list(), "dim": list()}
        for name in self.config.keys():
            mean = self.config[name]["mean"]
            cov = self.config[name]["cov"]
            dim = config[name]["dim"]
            num_samples = self.config[name]["num_samples"]

            # print(name)
            # print("Mean:\n", mean)
            # print("Covariance:\n", cov)

            x = self.rng.multivariate_normal(mean, cov, size=num_samples)
            data += [x[i, :] for i in range(x.shape[0])]

            obs["name"] += ([name,] * num_samples)
            obs["dim"] += ([dim,] * num_samples)
        
        # Get around 2D object issue
        self.data = np.empty(len(data), dtype=object)
        for i, arr in enumerate(data):
            self.data[i] = arr
        
        self.obs = pd.DataFrame(obs)

    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, key):
        
        if not isinstance(key, (int, np.int8, np.int16, np.int32, np.int64, torch.long)):
            raise ValueError(f"Dataset cannot be indexed with an index of type {type(key)}")

        if key < 0:
            key += len(self)

        if not (0 <= key < len(self)):
            raise ValueError(f"Index out of bounds")
        
        item = self.data[key]
        metadata = self.obs.iloc[key, :]

        x = (item, metadata)
        return x

class Simulated(BaseDataset):
    def __init__(
            self, 
            distribution_generator: str=None,
            distribution_generator_args: list=None,
            distribution_generator_kwargs: dict=None,
            data=None
        ):
        """
        Simulated dataset. This dataset is generated from a distribution generator.

        Parameters
        ----------
        distribution_generator : str
            The name of the distribution generator to use. Currently, the only
            supported generator is "GaussianGenerator".
        distribution_generator_args : list
            A list of arguments to pass to the distribution generator.
        distribution_generator_kwargs : dict
            A dictionary of keyword arguments to pass to the distribution generator.
        """

        if data is not None:
            self.data = data
        else:
            assert distribution_generator is not None

            if distribution_generator_args is None:
                distribution_generator_args = list()
            if distribution_generator_kwargs is None:
                distribution_generator_kwargs = dict()

            if distribution_generator == "GaussianGenerator":
                self.data = GaussianGenerator(*distribution_generator_args, **distribution_generator_kwargs)
            else:
                raise Exception

        self._index = np.arange(len(self.data.obs))
    
    @property
    def obs(self):
        return self.data.obs.iloc[self._index, :]
    
    def subset(self, mask: np.ndarray=None, inplace=True):
        if mask is not None and not isinstance(mask, (tuple, list, pd.Series, np.ndarray, torch.Tensor)):
            raise ValueError(f"Mask must be an iterable Boolean array")
        if mask is not None and len(mask) != len(self._index):
            raise ValueError(f"Mask must specify one value for each observation")

        mask = mask.astype(bool)

        if inplace:
            self._index = np.arange(len(self.data.obs)) if mask is None else self._index[mask]
            return
        
        sim = Simulated(data=self.data)
        sim._index = np.arange(len(sim.data.obs)) if mask is None else self._index[mask]
    
        return sim
    
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
