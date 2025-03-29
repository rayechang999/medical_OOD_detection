import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import OODDetector


class MSP(OODDetector):
    def __init__(
            self,
            threshold: float=None,
            threshold_quantile: float=None,
            method: str="max",
        ):
        """
        Maximum softmax probability (Hendrycks and Gimpel in ICLR 2017). Samples
        are classified as OOD if the maximum (over all classes) softmax probability predicted by the 
        model is below a set threshold.

        Parameters
        ----------
        threshold : float, default: None
            Threshold of the score below which observations are considered OOD.
        threshold_quantile : float, default: None
            Proportion of the training dataset classified as ID (used ot set the threshold).
        method : str, default: "max"
            Method to compute the score. By default, the method is "max", which 
            indicates that the OOD score should be computed as the maximum softmax
            probability, as in the original paper. Other methods include "entropy"
            in which the negative Shannon entropy of the softmax distribution is calculated.
        """
        if not sum([threshold is None, threshold_quantile is None]):
            raise ValueError(f"Exactly one of threshold or threshold_quantile must be given")
        
        if method not in ("max", "entropy"):
            raise ValueError(f"Unrecognized method: {method}")

        self.threshold = threshold
        self.threshold_quantile_score = 1 - threshold_quantile
        self.method = method
    
    def _score(self, softmax: torch.Tensor):
        """
        Computes the score from a tensor that has been softmaxed
        """
        if self.method == "max":
            score, _ = torch.max(softmax, dim=-1) # returns a tuple
        elif self.method == "entropy":
            score = -torch.sum(softmax * torch.log(softmax), dim=-1)
        else:
            raise Exception
        
        score = score.detach().cpu().numpy()

        return score

    def train(self, dataloader: torch.utils.data.DataLoader):
        """
        Trains the IsolationForest model using the latent representations obtained from the LSTM reservoir.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            A PyTorch DataLoader containing the training data. For this class and
            all other OOD detectors, the dataloader must load a pair (x, y). Only
            x will be used and y will be ignored. The x in the dataloader should be
            logits from a model.

        Returns
        -------
        dict
            A dictionary containing training metrics.
        """
        metrics = dict()
        
        out = torch.cat([x for x, y in dataloader], dim=0)
        
        softmax = F.softmax(out, dim=-1)
        score = self._score(softmax)

        if self.threshold is None:
            self.threshold = np.quantile(score, self.threshold_quantile_score)
            print(f"Computed {self.threshold} as the threshold: {round(100 * (score < self.threshold).mean(), 2)}% of training examples labeled as OOD")

        metrics["threshold"] = self.threshold
        metrics["threshold_quantile"] = 1 - self.threshold_quantile_score

        return metrics
    
    def predict(self, dataloader: torch.utils.data.DataLoader):
        """
        Predicts whether the inputs in the dataloader are out-of-distribution (OOD)
        using the trained IsolationForest model.

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

        softmax = F.softmax(out, dim=-1)
        score = self._score(softmax)

        ood = (score < self.threshold).astype(int)

        return ood

class MSPWithModel(OODDetector):
    def __init__(
            self,
            model: nn.Module,
            threshold: float=None,
            threshold_quantile: float=None,
            method: str="max",
        ):
        """
        Maximum softmax probability (Hendrycks and Gimpel in ICLR 2017). Samples
        are classified as OOD if the maximum (over all classes) softmax probability predicted by the 
        model is below a set threshold.

        Parameters
        ----------
        model : nn.Module
            Pre-trained multi-class classification model. Outputs of the model are
            expected to be logits.
        threshold : float, default: None
            Threshold of the score below which observations are considered OOD.
        threshold_quantile : float, default: None
            Proportion of training observations to classify as ID (to set threshold).
        method : str, default: "max"
            Method to compute the score. By default, the method is "max", which 
            indicates that the OOD score should be computed as the maximum softmax
            probability, as in the original paper. Other methods include "entropy"
            in which the Shannon entropy of the softmax distribution is calculated.
        """
        if not sum([threshold is None, threshold_quantile is None]):
            raise ValueError(f"Exactly one of threshold or threshold_quantile must be given")
        
        if method not in ("max", "entropy"):
            raise ValueError(f"Unrecognized method: {method}")

        self.model = model
        self.threshold = threshold
        self.threshold_quantile_score = 1 - threshold_quantile
        self.method = method
    
    def _score(self, softmax: torch.Tensor):
        """
        Computes the score from a tensor that has been softmaxed
        """
        if self.method == "max":
            score, _ = torch.max(softmax, dim=-1) # returns a tuple (max_values, indices)
        elif self.method == "entropy":
            score = -torch.sum(softmax * torch.log(softmax), dim=-1)
        else:
            raise Exception
        
        score = score.detach().cpu().numpy()

        return score
    
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
        
        softmax = F.softmax(out, dim=-1)
        score = self._score(softmax)

        if self.threshold is None:
            self.threshold = np.quantile(score, self.threshold_quantile_score)
            print(f"Computed {self.threshold} as the threshold: {round(100 * (score < self.threshold).mean(), 2)}% of training examples labeled as OOD")

        metrics["threshold"] = self.threshold
        metrics["threshold_quantile"] = 1 - self.threshold_quantile_score

        return metrics
    
    def predict(self, dataloader: torch.utils.data.DataLoader):
        """
        Predicts whether the inputs in the dataloader are out-of-distribution (OOD)
        using the trained IsolationForest model.

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

        softmax = F.softmax(out, dim=-1)
        score = self._score(softmax)

        ood = (score < self.threshold).astype(int)

        return ood