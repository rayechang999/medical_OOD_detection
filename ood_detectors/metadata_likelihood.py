import sklearn
import sklearn.ensemble

import scipy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from .base import OODDetector

import tqdm

class MetadataLikelihoodIndependent(OODDetector):
    def __init__(
            self,
            categorical_mask: list | tuple | np.ndarray | torch.Tensor,
            likelihood_threshold: float=None,
            likelihood_threshold_quantile: float=None,
            continuous_likelihood_method: str="norm",
            categorical_likelihood_method: str="Good-Turing",
            categorical_likelihood_method_params: dict=None
        ):
        """
        OOD detection using patient-level metadata.

        Parameters
        ----------
        categorical_mask : list | tuple | np.ndarray | torch.Tensor
            A list of booleans indicating whether each metadata feature is categorical.
            The order of the list should match the order of the metadata features.
            Inputs to this model are 1D tensors `x` in which `x[i]` is treated as
            categorical if and only if `categorical_mask[i] == True`; otherwise,
            it is treated as continuous.
        likelihood_threshold : float, optional
            A threshold for the likelihood. If specified, this value will be 
            used as the threshold for classifying inputs as OOD. Metadata that
            is less than this threshold will be considered OOD.
        likelihood_threshold_quantile : float, optional
            A quantile for the likelihood. If specified, this value will be
            used to compute likelihood_threshold. A likelihood_thresohld_quantile 
            of 0.05 indicates that the 5% quantile of the likelihoods evaluated 
            on the training dataset is what the likelihood_threshold is set to.
        continuous_likelihood_method : str, optional, default="norm"
            The method for computing probability densities of continuous variables.
            All distributions' parameters are learned by MLE.
            Options are:
            - "alpha": Use an alpha distribution to model the density.
            - "beta": Use a beta distribution to model the density.
            - "expon": Use an exponential distribution to model the density.
            - "gamma": Use a gamma distribution to model the density.
            - "norm": Use a normal distribution to model the density.
            - "lognorm": Use a log-normal distribution to model the density.
            - "t": Use a t-distribution to model the density.
            - "uniform": Use a uniform distribution to model the density.
        categorical_likelihood_method : str, optional
            A method for handling unseen categorical metadata values. Options are:
            - "constant": Set the probability of observing an unseen class to a constant value.
                Parameters:
                    - p: float, the constant probability of observing an unseen class.
            - "minimum": Use the minimum observed frequency to model the likelihood.
            - "minimum-constant": Set the probability of observing an unseen class
                to p(unseen) = min(smallest observed frequency, constant value p).
                Parameters:
                    - p_max: float, the maximum probability of observing an unseen class.
            - "entropy": Use the entropy of the distribution to model the likelihood.
                Here, the probability of observing an unseen class is modeled as
                p(unseen) = E / (alpha + E), where E is the entropy of the
                distribution (in the training set) and alpha is a non-negative
                hyperparameter that varies inversely with p(unseen).
                Parameters:
                    - alpha: float, non-negative, maybe try setting to 250-500ish?
            - "Good-Turing": The Good-Turing estimator, in which the probability
                of observing an unseen class is p(unseen) = f_1 / n, where f_1 
                is the number of classes that appear exactly once and n is the
                total number of observations.
            - "Chao": The Chao estimator, in which the probability of observing an
                an unseen class is p(unseen) = (f_1)^2 / (2 * f_2), where f_1 is
                the number of classes that appear exactly once and f_2 is the number
                of classes that appear exactly twice.
            - "Pitman-Yor": Pitman-Yor process in which the probability of observing
                an unseen class is p(unseen) = (theta + m * sigma) / (n + theta), 
                where n is the total number of observations and m is the total
                number of observed classes. The Dirichlet process is a special
                case of Pitman-Yor in which sigma = 0.
                Parameters:
                    - theta: float, the concentration hyperparameter
                    - sigma: float, the discount hyperparameter
                Note: The theta / (n + theta) term tends to dominate, and the
                resulting p(unseen) is usually close to this value.
        categorical_likelihood_method_params : dict, default=None
            Parameters (see above) for the categorical likelihood method selected.
        """
        # Check that exactly one of the thresholds is specified
        if sum([likelihood_threshold is not None, likelihood_threshold_quantile is not None]) != 1:
            raise ValueError("Exactly one of likelihood_threshold or likelihood_threshold_quantile must be specified.")

        self.categorical_mask = categorical_mask
        self.likelihood_threshold = likelihood_threshold
        self.likelihood_threshold_quantile = likelihood_threshold_quantile

        self.continuous_likelihood_method = continuous_likelihood_method
        self.categorical_likelihood_method = categorical_likelihood_method
        self.categorical_likelihood_method_params = categorical_likelihood_method_params

        dist_modules = {
            "alpha": scipy.stats.alpha,
            "beta": scipy.stats.beta,
            "expon": scipy.stats.expon,
            "gamma": scipy.stats.gamma,
            "norm": scipy.stats.norm,
            "lognorm": scipy.stats.lognorm,
            "t": scipy.stats.t,
            "uniform": scipy.stats.uniform
        }

        if self.continuous_likelihood_method not in dist_modules:
            raise ValueError(f"Unknown continuous_likelihood_method: {self.continuous_likelihood_method}")
        
        self.dist = dist_modules[self.continuous_likelihood_method]
    
    def train(self, dataloader: torch.utils.data.DataLoader):
        """
        Parameters
        ----------
        dataloader : DataLoader
            A DataLoader object containing the training data. The dataloader should
            be batched, and it must return a tuple (x, y), where x is a tensor of
            shape (batch_size, num_features) and y can be anything (it isn't used).
            The features in x must be in the same order as the categorical_mask.
        """

        self.params = {i: dict() for i in range(len(self.categorical_mask))}

        metrics = dict()

        x = torch.cat([x for x, y in dataloader], dim=0).detach().cpu().numpy()

        for i in range(len(self.categorical_mask)):
            if self.categorical_mask[i]:
                self._train_likelihood_categorical(i, x[:, i])
            else:
                self._train_likelihood_continuous(i, x[:, i])

        # Get threshold if not already
        if self.likelihood_threshold is None:
            likelihoods = np.array([self._predict_single(xi) for xi in x])

            self.likelihood_threshold = np.quantile(likelihoods, self.likelihood_threshold_quantile)
            print(f"Computed likelihood_threshold: {self.likelihood_threshold}, at which {(np.mean(likelihoods < self.likelihood_threshold) * 100):.4f}% of the training data is below this threshold.")
        
        metrics["likelihood_threshold"] = self.likelihood_threshold
        metrics["likelihood_threshold_quantile"] = self.likelihood_threshold_quantile

        return metrics
    
    def _train_likelihood_categorical(self, i: int, x: np.ndarray):
        """
        Sets self.params[i] keys 'n', 'values', 'counts', 'freqs', and 'p_unseen'.
        """
        x = np.squeeze(x)

        self.params[i]["n"] = len(x)
        self.params[i]["values"] = np.unique(x)
        self.params[i]["counts"] = {val: occurrences for val, occurrences in zip(*np.unique(x, return_counts=True))}
        self.params[i]["freqs"] = {val: occurrences / self.params[i]["n"] for val, occurrences in self.params[i]["counts"].items()}

        if self.categorical_likelihood_method == "constant":
            self.params[i]["p_unseen"] = self.categorical_likelihood_method_params["p"]
        elif self.categorical_likelihood_method == "minimum":
            self.params[i]["p_unseen"] = min(self.params[i]["freqs"].values())
        elif self.categorical_likelihood_method == "minimum-constant":
            min_freq = min(self.params[i]["freqs"].values())
            p = self.categorical_likelihood_method_params["p_max"]
            self.params[i]["p_unseen"] = min(min_freq, p)
        elif self.categorical_likelihood_method == "entropy":
            alpha = self.categorical_likelihood_method_params["alpha"]
            E = -np.sum([p * np.log(p) for p in self.params[i]["freqs"].values()]) if len(self.params[i]["values"]) > 1 else 0
            self.params[i]["p_unseen"] = E / (alpha + E)
        elif self.categorical_likelihood_method == "Good-Turing":
            n = self.params[i]["n"]
            f_1 = len([count for count in self.params[i]["counts"].values() if count == 1])
            self.params[i]["p_unseen"] = f_1 / n
        elif self.categorical_likelihood_method == "Chao":
            n = self.params[i]["n"]
            f_1 = len([count for count in self.params[i]["counts"].values() if count == 1])
            f_2 = len([count for count in self.params[i]["counts"].values() if count == 2])
            self.params[i]["p_unseen"] = (f_1 ** 2) / (2 * f_2)
        elif self.categorical_likelihood_method == "Pitman-Yor":
            n = self.params[i]["n"]
            m = len(self.params[i]["values"])
            theta = self.categorical_likelihood_method_params["theta"]
            sigma = self.categorical_likelihood_method_params["sigma"]
            self.params[i]["p_unseen"] = (theta + m * sigma) / (n + theta)
        else:
            raise ValueError(f"Unknown categorical_likelihood_method: {self.categorical_likelihood_method}")
        
        return
    
    def _train_likelihood_continuous(self, i: int, x: np.ndarray):

        x = np.squeeze(x)
        x = x[~np.isnan(x)]
        self.params[i]["params"] = self.dist.fit(x)

        return

    def _predict_likelihood_categorical(self, i: int, x: int) -> float:
        return self.params[i]["freqs"][x] if x in self.params[i]["values"] else self.params[i]["p_unseen"]

    def _predict_likelihood_continuous(self, i: int, x: float) -> float:
        shapes = self.params[i]["params"][:-2]
        loc = self.params[i]["params"][-2]
        scale = self.params[i]["params"][-1]

        return self.dist.pdf(x, *shapes, loc=loc, scale=scale)

    def _predict_single(self, x: torch.Tensor) -> float:
        """
        Gets the probability score of a single metadata tensor, which should *not*
        be batched.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape (num_features,). The features in x must be in the same order as the categorical_mask.
        
        Returns
        -------
        float
            The probability score of the input.
        """
        likelihood = 0.0
        for i, feature in enumerate(x):
            if self.categorical_mask[i]:
                likelihood += np.log(self._predict_likelihood_categorical(i, feature.item()))
            else:
                likelihood += np.log(self._predict_likelihood_continuous(i, feature.item()))
        
        return likelihood
    
    def predict(self, dataloader) -> np.ndarray:
        """
        
        Returns
        -------
        np.ndarray
            A numpy array of shape (num_samples,) containing the predicted labels.
            array[i] is 1 if dataloader.dataset[i] is OOD, and 0 otherwise.
        """

        x = torch.cat([x for x, y in dataloader], dim=0)
        likelihoods = np.array([self._predict_single(xi) for xi in x])

        labels = likelihoods < self.likelihood_threshold

        return labels.astype(int)
