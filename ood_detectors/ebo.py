import sklearn
import sklearn.ensemble

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from .base import OODDetector

import tqdm

class EBO(OODDetector):
    def __init__(
        self,
        temperature: float=1.0,
        threshold: float=None,
        threshold_quantile: float=None,
    ):
        """
        Energy-based OOD detection (Liu et al. in NeurIPS 2020).

        Parameters
        ----------
        temperature : float, default: 1.0
            Temperature for the algorithm.
        threshold : float, default: None
            Threshold for the algorithm. Either this or threshold_quantile must be provided.
        threshold_quantile : float, default: None
            Quantile for the threshold t* such that, when using t* as the threshold, 
            this quantile of training examples have a lower score than t* (and are
            thus identified as ID). For example, setting threshold_quantile to 0.95
            means that 95% of the training dataset's scores will fall below this 
            threshold and thus be classified as ID. Liu et al. note that, "in practice, we choose 
            the threshold using in-distribution data so that a high fraction of 
            inputs are correctly classified by the OOD detector."
        """
        if not sum([threshold is None, threshold_quantile is None]):
            raise ValueError(f"Exactly one of threshold or threshold_quantile must be given")

        self.temperature = temperature
        self.threshold = threshold
        self.threshold_quantile = threshold_quantile
    
    def train(self, dataloader: torch.utils.data.DataLoader):
        """
        Trains the IsolationForest model using the latent representations obtained from the LSTM reservoir.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            A PyTorch DataLoader containing the training data. For this class and
            all other OOD detectors, the dataloader must load a pair (x, y). Only
            x will be used and y will be ignored.

        Returns
        -------
        dict
            A dictionary containing training metrics.
        """
        metrics = dict()

        out = torch.cat([x for x, y in dataloader], dim=0)
        
        # Compute energy
        energy = -self.temperature * torch.logsumexp(out / self.temperature, dim=1)
        energy = energy.detach().cpu().numpy()

        score = -energy

        if self.threshold is None:
            self.threshold = np.quantile(score, self.threshold_quantile)
            print(f"Computed {self.threshold} as the threshold: {round(100 * (score > self.threshold).mean(), 2)}% of training examples labeled as OOD")

        metrics["threshold"] = self.threshold
        metrics["threshold_quantile"] = self.threshold_quantile

        return metrics
    
    def predict(self, dataloader: torch.utils.data.DataLoader):
        """
        Predicts whether the inputs in the dataloader are out-of-distribution (OOD)
        using the trained model.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            A PyTorch DataLoader containing the data to be predicted.

        Returns
        -------
        np.ndarray
            An array of binary labels indicating whether each input is OOD (1) or not (0).
        """
        if isinstance(dataloader.sampler, torch.utils.data.sampler.RandomSampler):
            raise Exception(f"The dataloader must load observations in a fixed order. Try initializing with shuffle=False.")

        out = torch.cat([x for x, y in dataloader], dim=0)

        # Compute energy
        energy = -self.temperature * torch.logsumexp(out / self.temperature, dim=1)
        energy = energy.detach().cpu().numpy()

        ood = (-energy > self.threshold).astype(int)

        return ood

class EBOWithModel(OODDetector):
    def __init__(
        self,
        model: nn.Module,
        temperature: float=1.0,
        threshold: float=None,
        threshold_quantile: float=None,
    ):
        """
        Energy-based OOD detection (Liu et al. in NeurIPS 2020).

        Parameters
        ----------
        model : nn.Module
            Pre-trained model to be used with this OOD detection algorithm. Outputs
            of the model are expected to be logits over a class distribution.
        temperature : float, default: 1.0
            Temperature for the algorithm.
        threshold : float, default: None
            Threshold for the algorithm. Either this or threshold_quantile must be provided.
        threshold_quantile : float, default: None
            Quantile for the threshold t* such that, when using t* as the threshold, 
            this quantile of training examples have a lower score than t* (and are
            thus identified as ID). For example, setting threshold_quantile to 0.95
            means that 95% of the training dataset's scores will fall below this 
            threshold and thus be classified as ID. Liu et al. note that, "in practice, we choose 
            the threshold using in-distribution data so that a high fraction of 
            inputs are correctly classified by the OOD detector."
        """
        if not sum([threshold is None, threshold_quantile is None]):
            raise ValueError(f"Exactly one of threshold or threshold_quantile must be given")
        
        self.model = model
        self.temperature = temperature
        self.threshold = threshold
        self.threshold_quantile = threshold_quantile
    
    def train(self, dataloader: torch.utils.data.DataLoader):
        """
        Trains the IsolationForest model using the latent representations obtained from the LSTM reservoir.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            A PyTorch DataLoader containing the training data. For this class and
            all other OOD detectors, the dataloader must load a pair (x, y). Only
            x will be used and y will be ignored.

        Returns
        -------
        dict
            A dictionary containing training metrics.
        """
        metrics = dict()

        self.model.eval()
        with torch.no_grad():
            out = torch.cat([self.model(x) for x, y in dataloader], dim=0)
        
        # Compute energy
        energy = -self.temperature * torch.logsumexp(out / self.temperature, dim=1)
        energy = energy.detach().cpu().numpy()

        score = -energy

        if self.threshold is None:
            self.threshold = np.quantile(score, self.threshold_quantile)
            print(f"Computed {self.threshold} as the threshold: {round(100 * (score > self.threshold).mean(), 2)}% of training examples labeled as OOD")

        metrics["threshold"] = self.threshold
        metrics["threshold_quantile"] = self.threshold_quantile

        return metrics
    
    def predict(self, dataloader: torch.utils.data.DataLoader):
        """
        Predicts whether the inputs in the dataloader are out-of-distribution (OOD)
        using the trained model.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            A PyTorch DataLoader containing the data to be predicted.

        Returns
        -------
        np.ndarray
            An array of binary labels indicating whether each input is OOD (1) or not (0).
        """
        if isinstance(dataloader.sampler, torch.utils.data.sampler.RandomSampler):
            raise Exception(f"The dataloader must load observations in a fixed order. Try initializing with shuffle=False.")

        self.model.eval()

        with torch.no_grad():
            out = torch.cat([self.model(x) for x, y in dataloader], dim=0)

        # Compute energy
        energy = -self.temperature * torch.logsumexp(out / self.temperature, dim=1)
        energy = energy.detach().cpu().numpy()

        ood = (-energy > self.threshold).astype(int)

        return ood