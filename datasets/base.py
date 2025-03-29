import abc
from abc import abstractmethod

import pandas as pd

import torch

class BaseDataset(abc.ABC):
    
    def __init__(self, *args, **kwargs):
        """
        Initializes the dataset. Common args include the local path to where 
        the data is stored, and possibly a path to where the metadata file is.
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the length of the object (number of observations), which may not
        be constant if the object has been subsetted.
        """
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> tuple: # returns (x, metadata)
        """
        Gets the idx-th example in the data. Returns a tuple (x, metadata), where 
        x is the requested data (an image, time series, etc.) and metadata is a 
        pd.Series which contains metadata for this particular observation.

        Args:
            idx (int): The integer index of the observation to be returned.
                Negative indices are supported
        
        Returns:
            (x, metadata): A tuple where x is the data for the idx-th observation
                (an image, time series, etc.) and metadata is a pandas Series 
                that contains the metadata for this observation.
        """
        pass
    
    def subset(self, mask=None, inplace: bool=True):
        """
        Subsets the dataset based on a Boolean mask of length len(self). If 
        mask[i] is True, then the i-th example will be included in the resulting
        subset, and otherwise it won't be. If mask is None, then all previous 
        subsetting operations are undone and the dataset is returned to its 
        original size. If inplace is True, then the object is subsetted in-place;
        otherwise, a new object is returned.

        Args:
            mask (np.ndarray | None): A Boolean mask where mask[i] == True if the i-th
                example is to be kept in the subset, otherwise False. If mask 
                is None, then all previous subsets are undone and the dataset is 
                returned to its original size
            inplace (bool): Whether or not the subsetting is to be performed in-place
                or whether a new object should be returned without modifying the
                original.
        
        Returns:
            A new dataset object that has been subsetted, or None if inplace == True.
        """
        pass

    @property
    def obs(self) -> pd.DataFrame:
        """
        Returns the observation-level metadata as a pandas DataFrame. Each row 
        is an observation (and the total number of rows in this DataFrame is
        equal to len(self)), and can have any number of columns.
        """
        pass

"""
BaseDatasets must define self.obs, a pd.DataFrame corresponding to the
current subset
"""

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        """
        Simple wrapper for transforming the outputs of a dataset.

        Examples:
            
            Transforming ISIC images:
                ```
                from .isic import ISIC

                raw_dataset = ISIC(...)
                transforms = torchvision.transforms.Compose(...)

                dataset = TorchDataset(raw_dataset, transforms)
                ```
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, key):
        x = self.dataset[key]
        
        if self.transform is not None:
            x = self.transform(x)
        
        return x