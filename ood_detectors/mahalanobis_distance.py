import sklearn
import sklearn.ensemble

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from .base import OODDetector


class MahalanobisDistance(OODDetector):
    def __init__(self, threshold: float=None, threshold_quantile: float=None):
        """
        Mahalanobis distance in the raw feature space.
        """
        if threshold is None and threshold_quantile is None:
            raise ValueError(f"One of threshold or threshold_quantile must be specified")
        elif threshold is not None and threshold_quantile is not None:
            print(f"WARNING: Both threshold and threshold_quantile are specified, using only threshold")
            threshold_quantile = None
        
        assert sum([threshold is None, threshold_quantile is None]) == 1

        self.threshold = threshold
        self.threshold_quantile = threshold_quantile

    def train(self, dataloader, sample=None) -> dict:

        X = torch.cat([x for x, y in dataloader], dim=0).detach().cpu().numpy() # risky
        Y = torch.cat([y for x, y in dataloader], dim=0).detach().cpu().numpy()

        self.labels = np.unique(Y)
        
        print(f"Inferred {len(self.labels)} distinct classes: {self.labels}")
        self.parameters = {label: {"mu": None, "cov": None, "cov_inv": None} for label in self.labels}

        for label in np.unique(Y):
            if sample is not None and sample < np.sum(Y == label):
                inds = np.random.choice(np.nonzero(Y == label)[0].squeeze(), size=sample, replace=False)
                x_label = X[inds]
            else:
                x_label = X[(Y == label).squeeze()]
        
            assert len(np.shape(x_label)) == 2

            self.parameters[label]["mu"] = np.mean(x_label, axis=0)
            self.parameters[label]["cov"] = np.cov(x_label, rowvar=False)

            try:
                self.parameters[label]["cov_inv"] = np.linalg.inv(self.parameters[label]["cov"])
            except Exception as e:
                print("Encountered exception e")
                print(f"Label: {label}")
                print(f"Shape: {np.shape(x_label)}")
                print(f"Mean: {self.parameters[label]['mu']}")
                print(f"Cov: {self.parameters[label]['cov']}")
                print(f"Shape of mean: {np.shape(self.parameters[label]['mu'])}")
                print(f"Shape of cov: {np.shape(self.parameters[label]['cov'])}")
                self.parameters[label]["cov_inv"] = np.linalg.inv(self.parameters[label]["cov"] + 1e-6 * np.eye(np.shape(x_label)[1]))
        
        if self.threshold_quantile is not None:
            print(f"Computing quantile threshold...")
            dist = np.array([self._predict_single(xi) for xi in X])
            self.threshold = float(np.quantile(dist, self.threshold_quantile))
            print(f"Computed {self.threshold} as the threshold: {round(100 * (dist > self.threshold).mean(), 2)}% of training examples labeled as OOD")

        return {"sample": sample, "threshold": self.threshold}

    def _predict_single(self, xi: np.ndarray):
        # Find the nearest class
        assert len(np.shape(xi)) == 1

        dist = list()
        for label in self.labels:
            diff = (xi - self.parameters[label]["mu"]).reshape(-1, 1) # column vector
            dist_label = diff.T @ self.parameters[label]["cov_inv"] @ diff
            dist.append(float(dist_label.squeeze()))

        return min(dist)
    
    def predict(self, dataloader):
        assert self.threshold is not None

        ood = list()
        for x, y in dataloader: # assumes x is batched
            ood += [self._predict_single(xi.detach().cpu().numpy()) for xi in x]

        return (np.array(ood) > self.threshold).astype(int)

class MahalanobisDistanceNN(OODDetector):
    def __init__(self, threshold: float=None, threshold_quantile: float=None):
        if sum([threshold is None, threshold_quantile is None]) != 1:
            raise ValueError(f"Either threshold or threshold_quantile must be specified, but not both")
        
        self.threshold = threshold
        self.threshold_quantile = threshold_quantile
    
    def train(self, dataloader, model: nn.Module, sample=None):
        """
        Args:
            sample (int | None): Number of samples, *per class*, to sample. If 
                None, then uses all available observations in each class. Default: None
        """
        self.model = model

        Z = torch.cat([self.model(x, return_latent=True) for x, y in dataloader], dim=0).detach().cpu().numpy() # risky
        Y = torch.cat([y for x, y in dataloader], dim=0).detach().cpu().numpy()

        self.labels = np.unique(Y)
        
        print(f"Inferred {len(self.labels)} distinct classes: {self.labels}")
        self.parameters = {label: {"mu": None, "cov": None, "cov_inv": None} for label in self.labels}

        for label in tqdm.tqdm(np.unique(Y), desc="Training Mahalnobis Distance NN"):
            if sample is not None and sample < np.sum(Y == label):
                inds = np.random.choice(np.nonzero(Y == label)[0].squeeze(), size=sample, replace=False)
                z_label = Z[inds]
            else:
                z_label = Z[(Y == label).squeeze()]
        
            assert len(np.shape(x_label)) == 2

            self.parameters[label]["mu"] = np.mean(z_label, axis=0)
            self.parameters[label]["cov"] = np.cov(z_label, rowvar=False)
            self.parameters[label]["cov_inv"] = np.linalg.inv(self.parameters[label]["cov"])
        
        if self.threshold_quantile is not None:
            print(f"Computing quantile threshold...")
            dist = np.array([self._predict_single(zi) for zi in Z])
            self.threshold = float(np.quantile(dist, self.threshold_quantile))
            print(f"Computed {self.threshold} as the threshold: {round(100 * (dist > self.threshold).mean(), 2)}% of training examples labeled as OOD")

        return {"sample": sample, "threshold": self.threshold}
    
    def _predict_single(self, zi: np.ndarray):
        # Find the nearest class
        assert len(np.shape(zi)) == 1

        dist = list()
        for label in self.labels:
            diff = (zi - self.parameters[label]["mu"]).reshape(-1, 1) # column vector
            dist_label = diff.T @ self.parameters[label]["cov_inv"] @ diff
            dist.append(float(dist_label.squeeze()))

        return min(dist)
    
    def predict(self, dataloader, device=None):
        assert self.threshold is not None

        ood = list()
        for x, y in dataloader:
            z = self.model(x, return_latent=True, device=device)
            ood += [self._predict_single(zi) for zi in z]
        
        return (np.array(ood) > self.threshold).astype(int)
            