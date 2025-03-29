import sklearn
import sklearn.ensemble

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from .base import OODDetector

class IsolationForest(OODDetector):
    def __init__(self):
        """
        Isolation Forest algorithm to detect OOD examples. This class
        is a wrapper around sklearn's IsolationForest class.
        """
        self.isolation_forest = sklearn.ensemble.IsolationForest()
    
    def train(self, dataloader, sample: int=None):
        if sample is not None:
            perm = np.random.permutation(len(dataloader.dataset)).astype(int)
            inds = perm[:sample]

            x = torch.cat([dataloader.dataset[i].unsqueeze(0) for i in inds], dim=0).detach().cpu().numpy()
        else:
            x = torch.cat([x for x, y in dataloader], dim=0).detach().cpu().numpy()

        self.isolation_forest.fit(x)

        metrics = dict()
        return metrics
    
    def predict(self, dataloader, return_score=False):

        if isinstance(dataloader.sampler, torch.utils.data.sampler.RandomSampler):
            raise Exception(f"The dataloader must load observations in a fixed order. Try initializing with shuffle=False.")
        
        x = torch.cat([x for x, y in iter(dataloader)], dim=0).detach().cpu().numpy()

        result = self.isolation_forest.score_samples(x) if return_score else (self.isolation_forest.predict(x) == -1).astype(int)
        return result

class IsolationForestWithModel(OODDetector):
    def __init__(self, model: nn.Module):
        """
        Isolation Forest algorithm to detect OOD examples. This class
        is a wrapper around sklearn's IsolationForest class.
        """
        self.model = model
        self.isolation_forest = sklearn.ensemble.IsolationForest()
    
    def train(self, dataloader, sample: int=None):
        self.model.eval()

        with torch.no_grad():
            if sample is not None:
                perm = np.random.permutation(len(dataloader.dataset)).astype(int)
                inds = perm[:sample]

                has_metadata = isinstance(dataloader.dataset[i], (list, tuple))
                raise Exception
                if not has_metadata:
                    x = torch.cat([self.model(dataloader.dataset[i].unsqueeze(0), return_latent=True) for i in inds], dim=0).detach().cpu().numpy()
                else:
                    x = torch.cat([self.model([elt.unsqueeze(0) for elt in dataloader.dataset[i]], return_latent=True) for i in inds], dim=0).detach().cpu().numpy()
            else:
                x = torch.cat([self.model(x, return_latent=True) for x, y in dataloader], dim=0).detach().cpu().numpy()

        self.isolation_forest.fit(x)

        metrics = dict()
        return metrics
    
    def predict(self, dataloader, return_score=False):

        if isinstance(dataloader.sampler, torch.utils.data.sampler.RandomSampler):
            raise Exception(f"The dataloader must load observations in a fixed order. Try initializing with shuffle=False.")

        self.model.eval()

        with torch.no_grad():
            x = torch.cat([self.model(x, return_latent=True) for x, y in dataloader], dim=0).detach().cpu().numpy()

        result = self.isolation_forest.score_samples(x) if return_score else (self.isolation_forest.predict(x) == -1).astype(int)
        return result
        
