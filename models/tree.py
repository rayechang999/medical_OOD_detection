import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import sklearn
import numpy as np
import lightgbm

import tqdm

from .base import BaseModel
from . import utils

class LightGBM(BaseModel):
    def __init__(self, param: dict):
        """
        LightGBM (https://lightgbm.readthedocs.io/en/stable/index.html)

        Parameters
        ----------
        param : dict
            A dictionary of parameters to be passed to lightgbm.train at
            training time. For example (from the Python Quick Start):
            {
                'num_leaves': 31,
                'objective': 'binary',
                'metric': ['auc', 'binary_logloss'],
            }
        """
        self.param = param

    def train(
            self,
            dataloader: torch.utils.data.DataLoader,
            num_round: int,
            val_dataloader: torch.utils.data.DataLoader=None,
            early_stopping: bool=False,
            save: str=None,
        ):
        """
        Train the LightGBM model.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            Dataloder outputting batched (x, y) examples to train on. Labels should
            be integers.
        num_round : int
            Number of rounds.
        val_dataloader : torch.utils.data.DataLoader, default: None
            Validation dataloader, optional.
        save : str, default: None
            Path to a directory to save results and model to.
        """
        if save is not None and not os.path.isdir(save):
            raise ValueError(f"save argument should be a directory")

        if early_stopping and val_dataloader is None:
            raise ValueError(f"Early stopping requires a validation dataset.")

        metrics = dict()

        self.early_stopping = early_stopping
        metrics["early_stopping"] = early_stopping
        metrics["param"] = self.param

        all_x = list()
        all_y = list()
        for x, y in dataloader:
            all_x.append(x)
            all_y.append(y)
        
        x_train = torch.cat(all_x, dim=0).detach().cpu().numpy()
        y_train = torch.cat(all_y, dim=0).detach().cpu().numpy()

        if val_dataloader is not None:
            all_x = list()
            all_y = list()
            for x, y in dataloader:
                all_x.append(x)
                all_y.append(y)
        
            x_val = torch.cat(all_x, dim=0).detach().cpu().numpy()
            y_val = torch.cat(all_y, dim=0).detach().cpu().numpy()

        train_data = lightgbm.Dataset(x_train, label=y_train)
        val_data = lightgbm.Dataset(x_val, label=y_val) if val_dataloader is not None else None

        if early_stopping:
            self.model = lightgbm.train(
                self.param,
                train_data,
                num_round,
                valid_sets = [val_data,],
                callbacks = [lightgbm.early_stopping(stopping_rounds=5)]
            )
            metrics["best_iteration"] = self.model.best_iteration
        else:
            self.model = lightgbm.train(
                self.param,
                train_data,
                num_round,
                valid_sets = [val_data,]
            )

        # Get metrics
        out, y = self.predict(dataloader, return_y=True, allow_random_sampler=True)
        metrics["train_accuracy"] = float((np.argmax(out.detach().cpu().numpy(), axis=1) == y).float().mean())
        metrics["train_AUROC"] = sklearn.metrics.roc_auc_score(
            y_true = nn.functional.one_hot(y, num_classes=len(np.unique(y))).detach().cpu().numpy(),
            y_score = out.detach().cpu().numpy()
        ) if len(np.unique(y)) > 1 else float("nan")

        if val_dataloader is not None:
            out, y = self.predict(val_dataloader, return_y=True, allow_random_sampler=True)
            metrics["val_accuracy"] = float((np.argmax(out.detach().cpu().numpy(), axis=1) == y).float().mean())
            metrics["val_AUROC"] = sklearn.metrics.roc_auc_score(
                y_true = nn.functional.one_hot(y, num_classes=len(np.unique(y))).detach().cpu().numpy(),
                y_score = out.detach().cpu().numpy()
            ) if len(np.unique(y)) > 1 else float("nan")

        if save is not None:
            model_path = os.path.join(save, "model.txt")
            self.model.save_model(model_path)
        
        return metrics

    def predict(
            self,
            dataloader: torch.utils.data.DataLoader,
            return_y: bool=False,
            allow_random_sampler: bool=False,
            device=None, # purely for compatibility, not used
        ):
        """
        Parameters
        ----------
        dataloader
            Dataloader that loads (x, y) pairs (y isn't used). Must be deterministic
            in loading and transforms.
        return_y : bool, default: False
            Whether to return the y labels alongside the predictions.

        Returns
        -------
        predictions : np.ndarray
            Predictions of the model
        y : np.ndarray, optional
            y labels if requested
        """
        if isinstance(dataloader.sampler, torch.utils.data.sampler.RandomSampler) and not allow_random_sampler:
            raise Exception(f"The dataloader must load observations in a fixed order. Try initializing with shuffle=False.")

        all_x = list()
        all_y = list()
        for x, y in dataloader:
            all_x.append(x)
            all_y.append(y)
        
        x = torch.cat(all_x, dim=0).detach().cpu().numpy()
        y = torch.cat(all_y, dim=0).detach().cpu()

        pred = self.model.predict(x, num_iteration=self.model.best_iteration if self.early_stopping else None)
        pred = np.vstack([1 - pred, pred]).T # reshape into (N, 2)
        pred = torch.tensor(pred)

        return (pred, y) if return_y else pred
