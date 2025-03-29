# OOD detection

This repository contains some code to train and evaluate several OOD detection 
methods on different datasets. Datasets are loaded through functions provided 
by `datasets`, ML models are defined in `models`, and OOD detectors are defined 
in `ood_detectors`. User-defined scripts that use these functions to run 
experiments are in `experiments`.

## Contents

- [Common Interfaces](#common-interfaces)
    - [datasets](#datasets)
    - [models](#models)
    - [ood_detectors](#ood_detectors)
- [Examples](#examples)

## Common Interfaces

### datasets

The `datasets` module provides code to load in the models. There's a class to 
load in each dataset (for example, the class `ISIC` for the ISIC dataset); indexing this 
object will return a tuple `(x, metadata)`, 
where `x` is the desired data (skin images for ISIC or voice recordings for 
Coswara) and `metadata` is a `pd.Series` containing metadata for this `x`. The 
full interface for a dataset is (see `datasets/base.py` for more details):

```python
class Dataset:
    
    def __init__(self, ...):
        """
        Initializes the dataset. Common args include the local path to where 
        the data is stored, and possibly a path to where the metadata file is.
        """
        pass
    
    def __len__(self):
        """
        Returns the length of the object (number of observations), which may not
        be constant if the object has been subsetted.
        """
        pass

    def __getitem__(self, idx: int) -> tuple:
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
```

`datasets.transforms` contains transform functions that help with transforming 
this `(x, metadata)` raw output from each dataset class into usable `(x, y)` 
training examples for downstream ML tasks. Each dataset has its own set of 
transforms defined (e.g. `datasets.transforms.coswara` contains Coswara-related
transforms).

`datasets` also contains a `TorchDataset` class which wraps around a dataset and 
a transform function, applying this transform function to the `(x, metadata)` 
pairs from the dataset before returning them, to make it easier to use these 
datasets with PyTorch.

### models

The `models` module defines various PyTorch models that can be trained on a 
particular dataset. Each class of models is defined in its own class (e.g.
`models.Conv2D` for 2D CNNs) and follows this interface:

```python
class Model:
    def __init__(self, ...):
        """
        Initializes the model. Usually, one of the arguments will be a string 
        `model`, which can be used to specify the exact architecture of this
        class of model. For example, when initializing `Conv2D(...)`, doing
        `Conv2D(model="M2")` will use the "M2" model (see models/conv2d.py for 
        the exact details of this architecture). Other common parameters include 
        the loss function type and learning rate.

        If you need access to the actual model (an object of type nn.Module), 
        it's usually stored in self.model.
        """
        pass
    
    def train(self, dataloader, ...) -> dict:
        """
        Trains the model on `(x, y)` pairs returned from a particular `dataloader`.

        Args:
            dataloader (torch.utils.data.DataLoader): A dataloader that returns
                (x, y) pairs.
        
        Returns:
            A dictionary of training metrics.
        """
        pass

    def predict(self, dataloader, return_y: bool=False, return_latent: bool=False, ...):
        """
        Predicts output using a given `dataloader` as input. This class assumes 
        that all of the raw data are too big to fit into memory at once, which 
        is why it uses a dataloader instead.

        Args:
            dataloader (torch.utils.data.DataLoader): A dataloader that returns 
                (x, y) pairs. Only x will be used as input to the model.
            return_y (bool): Whether or not to return a torch.Tensor that 
                contains all the `y` from the dataloader, concatenated in the 
                order that they were read. This can be useful for computing 
                downstream metrics (like AUROC), without having to load in 
                the entire dataset at once.
            return_latent (bool): Whether to return the neural network's intermediate
                layer representation of the data (return_latent == True) or return 
                the actual output of the neural network (return_latent == False).
                Returning intermediate/latent representation is sometimes useful
                for OOD detection algorithms.
        
        Returns:
            out: torch.Tensor or a tuple (out: torch.Tensor, y: torch.Tensor).
            If return_y is False, then just returns the neural network outputs 
            `out`, otherwise returns both the `out` and the `y`. If return_latent
            is True, then `out` represents the intermediate activations of the 
            neural network rather than the final output.
        """
        pass
```

### ood_detectors

The `ood_detectors` module is similar to `models` but it defines OOD detection 
algorithms instead of ML models. It follows a similar interface:

```python
class OODDetector:
    def __init__(self, ...):
        """
        Initializes the OOD detection algorithm. Common arguments include 
        the specific model architecture to be used (for DeepNearestNeighbors),
        or an nn.Module whose intermediate layer is to be extracted and used
        (for DeepNearestNeighborsWithModel), or the specific threshold to be 
        used for OOD detection.
        """
        pass
    
    def train(self, dataloader, ...) -> dict:
        """
        Trains the OOD detection algorithm using the specified `dataloader`.

        Args:
            dataloader (torch.utils.data.DataLoader): A dataloader that 
                returns (x, y) pairs (some OOD detection algorithms only 
                require x and no y, but yielding (x, y) pairs always is more 
                consistent across different OOD detection algorithms)
        
        Returns:
            A dictionary of training metrics.
        """
        pass
    
    def predict(self, dataloader, ...) -> np.ndarray:
        """
        Predicts which examples in the dataloader are OOD **in the order given**.
        This means that the dataloader should be initialized with shuffle=False so 
        that the OOD predictions are consistent across runs.

        NOTE: Please make sure that the dataloader has been initialized with
        shuffle=False, since this function returns predictions for each observation
        in the order that the observations were given.

        Args:
            dataloader (torch.utils.data.DataLoader): A dataloader that returns
                (x, y) pairs and which has been initialized with shuffle=False.
        
        Returns:
            A binary np.ndarray `arr` where arr[i] == 1 if the i-th example in 
            the dataloader is predicted to be OOD, otherwise arr[i] == 0.
        """
        pass
```

## Examples

User-defined experiment scripts are found in the `experiments` folder, but the 
code below provides some examples on how to train models and OOD detectors.

### Loading in a dataset

```python
import os
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision

import datasets
import models
import ood_detectors

dataset = datasets.ISIC(
    data_path = ".data/ISIC_2020",
    metadata_path = ".data/ISIC_2020/metadata.csv",
    verbose=True
)

print(f"Length of dataset: {len(dataset)}")

mask = np.random.choice([True, False], size=len(dataset))
train_dataset = dataset.subset(mask=mask, inplace=False)
test_dataset = dataset.subset(mask=~mask, inplace=False)

print(f"Length of training dataset: {len(train_dataset)}")

print(f"First training example: \n{train_dataset[0]}")
print(f"Training metadata: \n{train_dataset.obs.head()}")

# To use this dataset for training:

transforms = torchvision.transforms.Compose([
    datasets.transforms.isic.ApplyTo(torchvision.transforms.Resize((300, 300)), index=0),
    datasets.transforms.isic.ConvertYToTensor(
        metadata_to_y_func = lambda metadata: 0 if metadata["benign_malignant"] == "benign" else 1,
        y_dtype = torch.long
    ),
])
model_dataset = datasets.TorchDataset(
    train_dataset, 
    transform=model_transforms
)
model_dataloader = torch.utils.data.DataLoader(
    model_dataset, 
    batch_size = 64,
    shuffle = True,
)
```

### Training a model

```python
# Using the same imports as above

model = models.EfficientNet(model="M1", efficientnet="B3", loss="CrossEntropyLoss", ...)

train_metrics = model.train(model_dataloader, num_epochs=50, ...)

print(train_metrics)
```

### Training an OOD detector

```python
# Using the same imports as above

ood_detector = ood_detectors.IsolationForest(threshold_quantile=0.05)
ood_detector_train_metrics = ood_detector.train(ood_detector_dataloader, sample=None)

print(ood_detector_train_metrics)
```

### Predicting which samples are ID vs. OOD

```python
# Using the same imports as above

test_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle = False, # we have to define a new dataloader in order to set shuffle=False
    batch_size = 32,
)

ood = ood_detector.predict(test_dataloader)
```

### Evaluating model ID vs. OOD performance

```python
# Using the same imports as above

import sklearn
import torch
import torch.nn as nn

# Get AUROC on OOD samples
train_dataset.subset(None, inplace=True) # mask=None, so resets the subsetting
train_dataset.subset(ood.astype(bool), inplace=True)

out, y = model.predict(
    model_dataloader, # model_dataloader holds a reference to train_dataset, which is why we modified it in-place above so we don't have to define a new dataloader
    return_y = True, # so we can compute metrics later
)
ood_auroc = sklearn.metrics.roc_auc_score(
    y_true = nn.functional.one_hot(y, num_classes=2),
    y_score = nn.Softmax(dim=1)(out),
)
ood_acc = torch.mean(y == torch.argmax(out, dim=1))

# Get AUROC on ID samples
train_dataset.subset(None, inplace=True) # reset subsetting (restores train_dataset back to full length)
train_dataset.subset((1 - ood).astype(bool), inplace=True)

out, y = model.predict(
    model_dataloader,
    return_y = True,
)
id_auroc = sklearn.metrics.roc_auc_score(
    y_true = nn.functional.one_hot(y, num_classes=2),
    y_score = nn.Softmax(dim=1)(out),
)
id_acc = torch.mean(y == torch.argmax(out, dim=1))

print(f"Accuracy: {ood_acc} (OOD), {id_acc} (ID)")
print(f"AUROC: {ood_auroc} (OOD), {id_auroc} (ID)")
```
