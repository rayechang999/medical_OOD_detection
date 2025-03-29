# This is just benchmark_isic.py ported over to PDDB

import os
import argparse
import json
import sys

import re

import numpy as np
import pandas as pd
import sklearn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import datasets
import models
import ood_detectors

import utils

import shutil

import tqdm
import time

def train_mixup(
        model, 
        dataloader,
        val_dataloader, 
        loss_fn,
        optimizer, 
        epochs: int,
        alpha=1.0,
        device=None,
        save = None,
    ):
    """
    MixUp training. This implementation adapted from https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
    """
    def mixup_loss_fn(loss_fn, out, y_a, y_b, lambda_):
        return lambda_ * loss_fn(out, y_a) + (1 - lambda_) * loss_fn(out, y_b)

    model.to(device)

    metrics = {
        "train_loss": list(),
        "train_accuracy": list(),
        "train_AUROC": list(),
        "val_loss": list(),
        "val_accuracy": list(),
        "val_AUROC": list(),
    }

    for epoch in tqdm.tqdm(range(epochs), desc="Train MixUp"):

        model.train()

        epoch_loss = 0
        epoch_accuracy = 0
        all_outs = list()
        all_y = list()
        num_observations = 0

        for x, y in tqdm.tqdm(dataloader, desc=f"Model training (epoch {epoch})"):

            has_metadata = isinstance(x, (tuple, list))

            x = x.to(device) if not has_metadata else [tensor.to(device) for tensor in x]
            y = y.to(device)

            lambda_ = np.random.beta(alpha, alpha) if alpha > 0 else 1
            index = torch.randperm(x.size(0) if not has_metadata else x[0].size(0)).to(device)
            
            if has_metadata:
                for i in range(len(x)):
                    x[i] = lambda_ * x[i] + (1 - lambda_) * x[i][index, :]
                
                x_mixed = x
            else:
                x_mixed = lambda_ * x + (1 - lambda_) * x[index, :]
            
            y_a = y
            y_b = y[index]

            optimizer.zero_grad()

            out = model(x_mixed)
            loss = mixup_loss_fn(loss_fn, out, y_a, y_b, lambda_)

            loss.backward()
            optimizer.step()

            n = x.size(0) if isinstance(x, torch.Tensor) else x[0].size(0)

            epoch_loss += loss.detach().cpu().item() * n
            epoch_accuracy += float((torch.argmax(out, dim=1) == y).detach().cpu().float().mean() * n)
            
            all_outs.append(out.detach().cpu())
            all_y.append(y.detach().cpu())
            num_observations += float(n)

        epoch_loss /= num_observations
        epoch_accuracy /= num_observations

        all_outs = torch.cat(all_outs, dim=0)
        all_y = torch.cat(all_y).squeeze()

        out_probs = all_outs if torch.all(torch.abs(torch.sum(all_outs, dim=1) - 1.0) < 1e-4) else nn.Softmax(dim=1)(all_outs)
        epoch_AUROC = sklearn.metrics.roc_auc_score(
            y_true = nn.functional.one_hot(all_y, num_classes=len(torch.unique(all_y))).detach().cpu().numpy(),
            y_score = out_probs.detach().cpu().numpy()
        )

        print(epoch, epoch_loss, epoch_accuracy, epoch_AUROC)

        metrics["train_loss"].append(float(epoch_loss))
        metrics["train_accuracy"].append(float(epoch_accuracy))
        metrics["train_AUROC"].append(float(epoch_AUROC))

        if val_dataloader is not None:

            model.eval()

            with torch.no_grad():
                
                val_loss = 0
                val_accuracy = 0
                all_outs = list()
                all_y = list()
                num_observations = 0

                for x, y in val_dataloader:
                    has_metadata = isinstance(x, (tuple, list))

                    x = x.to(device) if not has_metadata else [tensor.to(device) for tensor in x]
                    y = y.to(device)

                    out = model(x)
                    loss = loss_fn(out, y)

                    n = x.size(0) if isinstance(x, torch.Tensor) else x[0].size(0)

                    val_loss += loss.detach().cpu().numpy() * n
                    val_accuracy += float((torch.argmax(out, dim=1) == y).detach().cpu().float().mean() * n)
                    
                    all_outs.append(out.detach().cpu())
                    all_y.append(y.detach().cpu())
                    num_observations += float(n)

                val_loss /= num_observations
                val_accuracy /= num_observations

                all_outs = torch.cat(all_outs, dim=0)
                all_y = torch.cat(all_y).squeeze()

                out_probs = all_outs if torch.all(torch.abs(torch.sum(all_outs, dim=1) - 1.0) < 1e-4) else nn.Softmax(dim=1)(all_outs)
                val_AUROC = sklearn.metrics.roc_auc_score(
                    y_true = nn.functional.one_hot(all_y, num_classes=len(torch.unique(all_y))).detach().cpu().numpy(),
                    y_score = out_probs
                )

            print(epoch, val_loss, val_accuracy, val_AUROC)
            
            metrics["val_loss"].append(float(val_loss))
            metrics["val_accuracy"].append(float(val_accuracy))
            metrics["val_AUROC"].append(float(val_AUROC))
    
    if save is not None:
        torch.save(model.state_dict(), os.path.join(save, "final.pt"))
    
    return metrics

class InstanceClassificationModel(nn.Module):
    def __init__(self, model, num_classes, x_example):
        super(InstanceClassificationModel, self).__init__()

        self.model = model

        with torch.no_grad():
            self.model.eval()
            in_features = self.model(x_example, return_latent=True).size(1)
        
        self.model.train()

        self.classifier = nn.Linear(in_features, out_features=num_classes)

    def forward(self, x):
        x = self.model(x, return_latent=True)
        x = self.classifier(x)

        return x

def train_instance_classification(
        model, 
        dataloader,
        optimizer, 
        epochs: int,
        device=None,
        save = None,
    ):
    """
    Instance classification, based on the ideas in Wu et al. 2018 (https://arxiv.org/abs/1805.01978)
    """
    class IndexedDataset:
        def __init__(self, dataset):
            self.dataset = dataset
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, key):
            x, _ = self.dataset[key]
            y = torch.tensor(key, dtype=torch.long)
            return x, y
    
    dataloader = torch.utils.data.DataLoader(
        IndexedDataset(dataloader.dataset),
        batch_size = 32,
        shuffle = True,
    )

    model = InstanceClassificationModel(model, len(dataloader.dataset), x_example=next(iter(dataloader))[0])
    loss_fn = nn.CrossEntropyLoss()

    model.to(device)

    metrics = {
        "train_loss": list(),
        "train_accuracy": list(),
        "train_AUROC": list(),
    }

    for epoch in tqdm.tqdm(range(epochs), desc="Train instance classification"):

        model.train()

        epoch_loss = 0
        epoch_accuracy = 0
        all_outs = list()
        all_y = list()
        num_observations = 0

        for x, y in tqdm.tqdm(dataloader, desc=f"Model training (epoch {epoch})"):

            has_metadata = isinstance(x, (tuple, list))

            x = x.to(device) if not has_metadata else [tensor.to(device) for tensor in x]
            y = y.to(device)

            optimizer.zero_grad()

            out = model(x)
            loss = loss_fn(out, y)

            loss.backward()
            optimizer.step()

            n = x.size(0) if isinstance(x, torch.Tensor) else x[0].size(0)

            epoch_loss += loss.detach().cpu().item() * n
            epoch_accuracy += float((torch.argmax(out, dim=1) == y).detach().cpu().float().mean() * n)
            
            all_outs.append(out.detach().cpu())
            all_y.append(y.detach().cpu())
            num_observations += float(n)

        epoch_loss /= num_observations
        epoch_accuracy /= num_observations

        all_outs = torch.cat(all_outs, dim=0)
        all_y = torch.cat(all_y).squeeze()

        out_probs = all_outs if torch.all(torch.abs(torch.sum(all_outs, dim=1) - 1.0) < 1e-4) else nn.Softmax(dim=1)(all_outs)
        epoch_AUROC = sklearn.metrics.roc_auc_score(
            y_true = nn.functional.one_hot(all_y, num_classes=len(torch.unique(all_y))).detach().cpu().numpy(),
            y_score = out_probs.detach().cpu().numpy()
        )

        print(epoch, epoch_loss, epoch_accuracy, epoch_AUROC)

        metrics["train_loss"].append(float(epoch_loss))
        metrics["train_accuracy"].append(float(epoch_accuracy))
        metrics["train_AUROC"].append(float(epoch_AUROC))

    if save is not None:
        torch.save(model.state_dict(), os.path.join(save, "final.pt"))

    return metrics

def train_logit_norm(
        model, 
        dataloader,
        optimizer,
        temperature: float,
        epochs: int,
        val_dataloader = None,
        crossentropy_args: list=None,
        crossentropy_kwargs: dict=None,
        device=None,
        save = None,
    ):
    """
    
    """
    if crossentropy_args is None:
        crossentropy_args = list()
    if crossentropy_kwargs is None:
        crossentropy_kwargs = dict()

    loss_fn = models.utils.LogitNormLoss(
        temperature,
        *crossentropy_args,
        **crossentropy_kwargs,
    )

    model.to(device)

    metrics = {
        "train_loss": list(),
        "train_accuracy": list(),
        "train_AUROC": list(),
        "val_loss": list(),
        "val_accuracy": list(),
        "val_AUROC": list(),
    }

    for epoch in tqdm.tqdm(range(epochs), desc="Train logit norm"):

        model.train()

        epoch_loss = 0
        epoch_accuracy = 0
        all_outs = list()
        all_y = list()
        num_observations = 0

        for x, y in tqdm.tqdm(dataloader, desc=f"Model training (epoch {epoch})"):

            has_metadata = isinstance(x, (tuple, list))

            x = x.to(device) if not has_metadata else [tensor.to(device) for tensor in x]
            y = y.to(device)

            optimizer.zero_grad()

            out = model(x)
            out = F.normalize(out, p=2, dim=1) / temperature
            loss = loss_fn(out, y)

            loss.backward()
            optimizer.step()

            n = x.size(0) if isinstance(x, torch.Tensor) else x[0].size(0)

            epoch_loss += loss.detach().cpu().item() * n
            epoch_accuracy += float((torch.argmax(out, dim=1) == y).detach().cpu().float().mean() * n)
            
            all_outs.append(out.detach().cpu())
            all_y.append(y.detach().cpu())
            num_observations += float(n)

        epoch_loss /= num_observations
        epoch_accuracy /= num_observations

        all_outs = torch.cat(all_outs, dim=0)
        all_y = torch.cat(all_y).squeeze()

        out_probs = all_outs if torch.all(torch.abs(torch.sum(all_outs, dim=1) - 1.0) < 1e-4) else nn.Softmax(dim=1)(all_outs)
        epoch_AUROC = sklearn.metrics.roc_auc_score(
            y_true = nn.functional.one_hot(all_y, num_classes=len(torch.unique(all_y))).detach().cpu().numpy(),
            y_score = out_probs.detach().cpu().numpy()
        )

        print(epoch, epoch_loss, epoch_accuracy, epoch_AUROC)

        metrics["train_loss"].append(float(epoch_loss))
        metrics["train_accuracy"].append(float(epoch_accuracy))
        metrics["train_AUROC"].append(float(epoch_AUROC))
        
        if val_dataloader is not None:

            model.eval()

            with torch.no_grad():
                
                val_loss = 0
                val_accuracy = 0
                all_outs = list()
                all_y = list()
                num_observations = 0

                for x, y in val_dataloader:
                    has_metadata = isinstance(x, (tuple, list))

                    x = x.to(device) if not has_metadata else [tensor.to(device) for tensor in x]
                    y = y.to(device)

                    out = model(x)
                    out = F.normalize(out, p=2, dim=1) / temperature
                    loss = loss_fn(out, y)

                    n = x.size(0) if isinstance(x, torch.Tensor) else x[0].size(0)

                    val_loss += loss.detach().cpu().numpy() * n
                    val_accuracy += float((torch.argmax(out, dim=1) == y).detach().cpu().float().mean() * n)
                    
                    all_outs.append(out.detach().cpu())
                    all_y.append(y.detach().cpu())
                    num_observations += float(n)

                val_loss /= num_observations
                val_accuracy /= num_observations

                all_outs = torch.cat(all_outs, dim=0).cpu()
                all_y = torch.cat(all_y).squeeze().cpu()

                out_probs = all_outs if torch.all(torch.abs(torch.sum(all_outs, dim=1) - 1.0) < 1e-4) else nn.Softmax(dim=1)(all_outs)
                val_AUROC = sklearn.metrics.roc_auc_score(
                    y_true = nn.functional.one_hot(all_y, num_classes=len(torch.unique(all_y))).detach().cpu().numpy(),
                    y_score = out_probs
                )

            print(epoch, val_loss, val_accuracy, val_AUROC)
            
            metrics["val_loss"].append(float(val_loss))
            metrics["val_accuracy"].append(float(val_accuracy))
            metrics["val_AUROC"].append(float(val_AUROC))
    
    if save is not None:
        torch.save(model.state_dict(), os.path.join(save, "final.pt"))

    return metrics

class PretrainedOODEnsemble:
    def __init__(self, ood_detectors: list[str], predictions: list[np.ndarray], method="intersection"):
        """
        Parameters
        ----------
        ood_detectors : list[str]
            List of names of OOD detectors
        predictions : list[np.ndarray]
            List of binary predictions of each OOD detector
        method : str
            Take final OOD prediction as intersection, union, or majority vote of individual predictions
        """
        if method not in ["intersection", "union", "majority"]:
            raise ValueError("Invalid method. Choose from 'intersection', 'union', or 'majority'.")
        if len(ood_detectors) == 0:
            raise ValueError("Must provide at least one OOD detector")
        if len(predictions) == 0:
            raise ValueError("Must provide at least one prediction")
        if len(ood_detectors) != len(predictions):
            raise ValueError("Length of OOD detectors must match length of predictions")

        self.ood_detectors = ood_detectors
        self.predictions = predictions
        self.method = method
    
    def predict(self) -> np.ndarray:
        predictions = self.predictions

        if self.method == "intersection":
            return np.all(predictions, axis=0).astype(int)
        elif self.method == "union":
            return np.any(predictions, axis=0).astype(int)
        elif self.method == "majority":
            return (np.sum(predictions, axis=0) > len(self.ood_detectors) / 2).astype(int)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--save", type=str, required=True,
        help="Path to the directory in which outputs will be saved")
    parser.add_argument("--load", type=str, default=None,
        help="Path to the directory with saved outputs from which this run should be resumed. If None, runs this script from the beginning. Only has an effect if --resume is also provided.")
    parser.add_argument("--resume", type=str, default=None, choices=[None, "train_model", "model_embedding", "train_ood"],
        help="The part of the script which we should start at. Requires that --load is also provided. Options: 'train_ood' starts at the training OOD detectors step")
    # parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    RESUME_MAPPER = {
        None: 0,
        "train_model": 1, # same as None basically
        "model_embedding": 2,
        "train_ood": 3,
        "bootstrap": 4, # not implemented, OOD detectors will be trained anyways
    }
    RESUME = RESUME_MAPPER[args.resume]

    # ---- DEFINE CONFIG ----
    GLOBAL_THRESHOLD_QUANTILE = 0.95
    CONFIG = {
        "seed": 2024,
        "device": "cuda:0",
        "num_bootstrap_replicates": 1000,
        "resume": args.resume,
        "dataset": {
            "path": "/path/to/Parkinson-new",
        },
        "ensemble_sizes": [3, 5], # ensemble_size == 1 is included by default
        "model": {
            "name": "Conv2D",
            "params": {
                "model": "GuanVersion1_4000",
                "batch_size": 128,
                "shuffle": True,
                "epochs": 50,
                "train_frac": 0.80,
            }
        },
        "ood_detectors": {
            "ensemble_method": "intersection",
            "knn_k": 25,
            "knn_threshold": None,
            "knn_threshold_quantile": GLOBAL_THRESHOLD_QUANTILE,
            "logit_norm_temperature": 0.04, # default for CIFAR-10 in Wei et al. 2022
            "mahalanobis_threshold": None,
            "mahalanobis_threshold_quantile": GLOBAL_THRESHOLD_QUANTILE,
            "ebo_threshold": None,
            "ebo_threshold_quantile": GLOBAL_THRESHOLD_QUANTILE,
            "msp_threshold": None,
            "msp_threshold_quantile": GLOBAL_THRESHOLD_QUANTILE,
            "msp_entropy_threshold": None,
            "msp_entropy_threshold_quantile": GLOBAL_THRESHOLD_QUANTILE,
        },
        "comments": ""
    }
    
    # Create folders
    if os.path.exists(args.save):
        raise FileExistsError(f"ERROR: path '{args.save}' already exists")
    os.makedirs(args.save)

    if RESUME > 0:
        for folder in tqdm.tqdm(os.listdir(args.load), "Copy results from load folder"):
            path = os.path.join(args.load, folder)
            if not os.path.isdir(path):
                continue

            print(f"Copying {folder} to new save...")
            shutil.copytree(path, os.path.join(args.save, folder))

    with open(__file__, "r") as f:
        current_script_file_contents = f.read()
    
    with open(os.path.join(args.save, "run_script.py"), "w") as f:
        f.write(current_script_file_contents)
    
    STDOUT_WRITE_FILE = open(os.path.join(args.save, "stdout.txt"), "w")
    dual_writer_stdout = utils.DualWriter(sys.stdout, STDOUT_WRITE_FILE)
    sys.stdout = dual_writer_stdout

    STDERR_WRITE_FILE = open(os.path.join(args.save, "stderr.txt"), "w")
    dual_writer_stderr = utils.DualWriter(sys.stderr, STDERR_WRITE_FILE)
    sys.stderr = dual_writer_stderr

    # ---- Handle arguments ----

    if CONFIG["seed"] is not None:
        print(f"Using seed: {CONFIG['seed']}")
        np.random.seed(CONFIG["seed"])
        torch.manual_seed(CONFIG["seed"])
        torch.cuda.manual_seed_all(CONFIG["seed"])

    print(f"Using config:\n{json.dumps(CONFIG, indent=2)}")

    with open(os.path.join(args.save, "CONFIG.json"), "w") as f:
        f.write(json.dumps(CONFIG, indent=2))

    # ---- Load in dataset ----

    print("Loading in the dataset")
    dataset = datasets.PDDB(
        path = CONFIG["dataset"]["path"],
        verbose=True
    )
    # Subsetting code from Josh
    rng = np.random.default_rng(seed=0)

    print(len((dataset.obs["professional-diagnosis"].astype(float) == 0).squeeze()))
    print(len((dataset.obs["professional-diagnosis"].astype(float) == 1).squeeze()))

    subsample_single_size = min(
        np.sum((dataset.obs["professional-diagnosis"].astype(float) == 0).squeeze()),
        np.sum((dataset.obs["professional-diagnosis"].astype(float) == 1).squeeze()),
    )

    print(f"Subsampling {subsample_single_size} positives and {subsample_single_size} negatives...")

    subsample_negative = rng.choice(np.argwhere(dataset.obs["professional-diagnosis"].astype(float) == 0).squeeze(), size=subsample_single_size, replace=False)
    subsample_positive = rng.choice(np.argwhere(dataset.obs["professional-diagnosis"].astype(float) == 1).squeeze(), size=subsample_single_size, replace=False)
    subsample = np.zeros(len(dataset)).astype(bool)
    subsample[subsample_negative] = True
    subsample[subsample_positive] = True
    
    # Filter out nan values
    GLOBAL_SUBSET = np.ones(len(dataset)).astype(bool)
    GLOBAL_SUBSET = (
        GLOBAL_SUBSET
        & (~pd.isna(dataset.obs["professional-diagnosis"]))
        & (dataset.obs["data::type"] == "rest_50Hz")
        & subsample
    )
    GLOBAL_SUBSET = np.array(GLOBAL_SUBSET, dtype=bool)

    print(f"Number of observations in dataset after global subsetting: {sum(GLOBAL_SUBSET)}")
    dataset.subset(GLOBAL_SUBSET, inplace=True)

    # Subset by individual
    #early_individuals = dataset.obs["healthCode"].unique()
    early_individuals = dataset.obs.loc[dataset.obs["age"] > 45, "healthCode"].unique()

    # Randomly shuffle the early_individuals
    shuffled_individuals = rng.permutation(early_individuals)

    # Calculate the split index (80% for training, 20% for testing)
    split_idx = int(0.8 * len(shuffled_individuals))

    # Split the shuffled individuals into train and test sets
    train_individuals = shuffled_individuals[:split_idx]
    test_individuals = shuffled_individuals[split_idx:]
    ALL_TRAIN_SUBSET = dataset.obs["healthCode"].isin(train_individuals).to_numpy()
    TEST_SUBSET = ~ALL_TRAIN_SUBSET

    temp_dataset = dataset.subset(ALL_TRAIN_SUBSET, inplace=False)
    #train_individuals = rng.choice(dataset.obs["healthCode"].to_numpy(), size=int(len(temp_dataset) * CONFIG["model"]["params"]["train_frac"]), replace=False)
    val_split_idx = int(CONFIG["model"]["params"]["train_frac"] * len(train_individuals))
    print(len(train_individuals))
    shuffled_train_individuals = rng.permutation(train_individuals)

    final_train_individuals = train_individuals[:val_split_idx]
    val_individuals = train_individuals[val_split_idx:]
    TRAIN_SUBSET = temp_dataset.obs["healthCode"].isin(final_train_individuals).to_numpy()
    #VAL_SUBSET = dataset.obs["healthCode"].isin(val_individuals).to_numpy()
    #TRAIN_SUBSET = temp_dataset.obs["healthCode"].isin(train_individuals).to_numpy()
    VAL_SUBSET = ~TRAIN_SUBSET

    # Subset and prepare for training (from ISIC)
    train_dataset = temp_dataset.subset(TRAIN_SUBSET, inplace=False) # use temp_dataset because that's how TRAIN_SUBSET is defined
    val_dataset = temp_dataset.subset(VAL_SUBSET, inplace=False)
    test_dataset = dataset.subset(TEST_SUBSET, inplace=False)
    dummy_test_dataset = dataset.subset(TEST_SUBSET, inplace=False) # for use at very end
    
    dataset.subset(ALL_TRAIN_SUBSET)
    dataset.subset(TRAIN_SUBSET)

    print(f"Using {len(train_dataset)} observations for training, {len(val_dataset)} observations for validation, and {sum(TEST_SUBSET)} observations for testing")
    
    # Save train dataset
    os.makedirs(os.path.join(args.save, "data"), exist_ok=True)
    train_dataset.obs.to_csv(os.path.join(args.save, "data", "train.csv"), index=True)
    val_dataset.obs.to_csv(os.path.join(args.save, "data", "val.csv"), index=True)
    test_dataset.obs.to_csv(os.path.join(args.save, "data", "test.csv"), index=True)
    
    # ---- Define general utility variables ----

    DEVICE = torch.device(CONFIG["device"] if CONFIG["device"] is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Using device: {DEVICE}")

    METADATA_TO_Y_FUNC = lambda metadata: int(metadata["professional-diagnosis"])

    base_transforms = torchvision.transforms.Compose([
        datasets.transforms.pddb.ApplyTo(
            datasets.transforms.pddb.Transpose(),
            index = 0,
        ),
    ])

    # ---- Load in the model ----

    model_transforms = torchvision.transforms.Compose([
        base_transforms,
        datasets.transforms.pddb.ApplyTo(
            datasets.transforms.pddb.AxisNormalization(axis=1),
            index = 0
        ),
        datasets.transforms.pddb.ApplyTo(
            datasets.transforms.pddb.TemporalRescaling(scale_factor_min=0.8, scale_factor_max=1.2),
            index = 0,
        ),
        datasets.transforms.pddb.ApplyTo(
            datasets.transforms.pddb.RandomRotation(),
            index = 0,
        ),
        datasets.transforms.pddb.ApplyTo(
            datasets.transforms.pddb.MagnitudeRescaling(scale_factor_min=0.8, scale_factor_max=1.2),
            index = 0,
        ),
        datasets.transforms.pddb.ApplyTo(
            datasets.transforms.pddb.SetSize(dim=1, size=4000),
            index = 0,
        ),
        datasets.transforms.pddb.ApplyTo(
            datasets.transforms.pddb.AsTensor(),
            index = 0
        ),
        datasets.transforms.pddb.ApplyTo(
            datasets.transforms.pddb.ConvertYToTensor(
                metadata_to_y_func = METADATA_TO_Y_FUNC,
                y_dtype = torch.long
            ),
            index = 1
        ),
    ])
    model_transforms_deterministic = torchvision.transforms.Compose([
        base_transforms,
        datasets.transforms.pddb.ApplyTo(
            datasets.transforms.pddb.AxisNormalization(axis=1),
            index = 0
        ),
        datasets.transforms.pddb.ApplyTo(
            datasets.transforms.pddb.SetSize(dim=1, size=4000),
            index = 0,
        ),
        datasets.transforms.pddb.ApplyTo(
            datasets.transforms.pddb.AsTensor(),
            index = 0
        ),
        datasets.transforms.pddb.ApplyTo(
            datasets.transforms.pddb.ConvertYToTensor(
                metadata_to_y_func = METADATA_TO_Y_FUNC,
                y_dtype = torch.long
            ),
            index = 1
        ),
    ])
    model_train_dataset = datasets.TorchDataset(
        train_dataset, 
        transform=model_transforms
    )
    model_train_dataloader = torch.utils.data.DataLoader(
        model_train_dataset, 
        batch_size=CONFIG["model"]["params"]["batch_size"],
        shuffle=CONFIG["model"]["params"]["shuffle"],
    )
    model_train_dataloader_deterministic = torch.utils.data.DataLoader(
        datasets.TorchDataset(
            train_dataset,
            transform = model_transforms_deterministic,
        ),
        batch_size = CONFIG["model"]["params"]["batch_size"],
        shuffle = CONFIG["model"]["params"]["shuffle"],
    )
    model_train_dataloader_fixed = torch.utils.data.DataLoader( # for use with OOD detectors
        datasets.TorchDataset(
            train_dataset,
            transform = model_transforms_deterministic,
        ),
        batch_size = CONFIG["model"]["params"]["batch_size"],
        shuffle = False,
    )
    model_val_dataset = datasets.TorchDataset(
        val_dataset,
        transform=model_transforms_deterministic
    )
    model_val_dataloader = torch.utils.data.DataLoader(
        model_val_dataset,
        batch_size = CONFIG["model"]["params"]["batch_size"],
        shuffle = False,
    )
    model_test_dataset = datasets.TorchDataset(
        test_dataset,
        transform = model_transforms_deterministic,
    )
    model_test_dataloader = torch.utils.data.DataLoader(
        model_test_dataset,
        batch_size = CONFIG["model"]["params"]["batch_size"],
        shuffle = False,
    )

    # label_factor, label_counts = np.unique(np.concatenate([y.cpu().numpy() for x, y in iter(model_dataloader)], axis=0), return_counts=True)
    label_factor, label_counts = np.unique([METADATA_TO_Y_FUNC(train_dataset.obs.iloc[i, :]) for i in range(len(train_dataset))], return_counts=True)
    print(f"Unique labels and counts: {label_factor} {label_counts} {np.round(label_counts / np.sum(label_counts), 4)}")
    
    model_train_weights = torch.tensor(np.sum(label_counts) / label_counts, dtype=torch.get_default_dtype(), device=DEVICE)
    print("weights", model_train_weights)

    dataset.subset()
    dataset.subset(GLOBAL_SUBSET)
    dataset.subset(ALL_TRAIN_SUBSET)
    dataset.subset(TRAIN_SUBSET)

    # Train model
    print("Training model")
    os.makedirs(os.path.join(args.save, "models"), exist_ok=True)

    # ---- Train latent models for latent-based OOD detection algorithms ----

    MODEL_KEYS = list()
    for model_type in ["base", "MixUp", "instance_classification", "logit_norm"]:
        MODEL_KEYS.append(model_type)
        for ensemble_size in CONFIG["ensemble_sizes"]:
            for i in range(ensemble_size):
                MODEL_KEYS.append(f"{model_type}_ensemble_{ensemble_size}_{i}")
    
    print(f"Using these model keys: {MODEL_KEYS}")

    x, y = next(iter(model_train_dataloader_deterministic))

    nets = {key: {"model": models.Conv2D(
        model = CONFIG['model']['params']['model'],
        input_shape = x.shape[1:],
        out_features = len(label_factor),
        loss = "CrossEntropyLoss",
        loss_kwargs = {"weight": torch.tensor(np.sum(label_counts) / label_counts, dtype=torch.get_default_dtype(), device=DEVICE)}
    )} for key in MODEL_KEYS}

    if RESUME is not None and 1 < RESUME:
        # Then load the models from pre-trained
        for key in tqdm.tqdm(nets.keys(), desc="Load models"):
            if "instance_classification" in key:
                temp_model = InstanceClassificationModel(
                    model = nets[key]["model"].model,
                    num_classes = len(model_train_dataloader_deterministic.dataset),
                    x_example = next(iter(model_train_dataloader_deterministic))[0],
                )
                temp_model.load_state_dict(torch.load(os.path.join(args.save, "models", key, "final.pt")))

                nets[key]["model"].model = temp_model.model
            else:
                model_state_dict = torch.load(os.path.join(args.save, "models", key, "final.pt"))
                nets[key]["model"].model.load_state_dict(model_state_dict)
    else:
        # Train models
        for key in tqdm.tqdm(nets.keys(), desc="Train models"):
            model_save_path = os.path.join(args.save, "models", key)
            os.makedirs(model_save_path, exist_ok=True)

            if "base" in key:
                train_metrics = nets["base"]["model"].train(
                    dataloader = model_train_dataloader,
                    val_dataloader = model_val_dataloader,
                    num_epochs = CONFIG["model"]["params"]["epochs"],
                    device = DEVICE,
                    save = model_save_path,
                )
            elif "logit_norm" in key:
                train_metrics = train_logit_norm(
                    model = nets[key]["model"].model,
                    dataloader = model_train_dataloader_deterministic,
                    optimizer = nets[key]["model"].optimizer,
                    temperature = CONFIG["ood_detectors"]["logit_norm_temperature"],
                    epochs = CONFIG["model"]["params"]["epochs"],
                    val_dataloader = model_val_dataloader,
                    crossentropy_kwargs = {"weight": torch.tensor(np.sum(label_counts) / label_counts, dtype=torch.get_default_dtype(), device=DEVICE)},
                    device=DEVICE,
                    save = model_save_path,
                )
            elif "MixUp" in key:
                # MixUp: Zhang et al. in ICLR, 2018 (https://arxiv.org/abs/1710.09412)
                train_metrics = train_mixup(
                    model = nets[key]["model"].model,
                    dataloader = model_train_dataloader_deterministic,
                    val_dataloader = model_val_dataloader,
                    loss_fn = nets[key]["model"].loss_fn,
                    optimizer = nets[key]["model"].optimizer,
                    epochs = CONFIG["model"]["params"]["epochs"],
                    alpha=1.0,
                    device=DEVICE,
                    save = model_save_path,
                )
            elif "instance_classification" in key:
                # Instance classification
                train_metrics = train_instance_classification(
                    model = nets[key]["model"].model,
                    dataloader = model_train_dataloader_deterministic,
                    optimizer = nets[key]["model"].optimizer,
                    epochs = CONFIG["model"]["params"]["epochs"],
                    device=DEVICE,
                    save = model_save_path,
                )
            else:
                raise ValueError(key)

            with open(os.path.join(model_save_path, "train_metrics.json"), "w") as f:
                f.write(json.dumps(train_metrics, indent=2))

    # ---- OOD detection: Latent-based (make the predictions here) -----
    detectors = dict()

    class MatrixDataset:
        def __init__(self, mat: torch.Tensor, y: torch.Tensor=None):
            self.mat = mat
            self.y = y

            if y is not None:
                assert len(y) == len(mat)
        
        def __len__(self):
            return np.shape(self.mat)[0]
        
        def __getitem__(self, index):
            return self.mat[index, :], (self.y[index] if self.y is not None else torch.tensor(0, dtype=torch.long))
    
    if RESUME is not None and 2 < RESUME:
        for key in tqdm.tqdm(nets.keys(), desc="Load model embeddings"):
            nets[key]["model"].model.eval()

            model_embedding_save_path = os.path.join(args.save, "models", key)
            train_embedding = torch.load(os.path.join(model_embedding_save_path, "train_embedding.pt"))
            train_y = torch.load(os.path.join(model_embedding_save_path, "train_y.pt"))
            test_embedding = torch.load(os.path.join(model_embedding_save_path, "test_embedding.pt"))

            nets[key]["train_embedding"] = train_embedding
            nets[key]["train_embedding_dataloader"] = torch.utils.data.DataLoader(
                MatrixDataset(nets[key]["train_embedding"], train_y),
                batch_size = 32,
                shuffle = False,
            )
            nets[key]["test_embedding"] = test_embedding
            nets[key]["test_embedding_dataloader"] = torch.utils.data.DataLoader(
                MatrixDataset(nets[key]["test_embedding"], y=None),
                batch_size = 32,
                shuffle = False,
            )
    else:
        for key in tqdm.tqdm(nets.keys(), "Compute model embeddings"):
            nets[key]["model"].model.eval()
            nets[key]["model"].model.to(DEVICE)
            
            # Train embeddings
            with torch.no_grad():
                embedding = list()
                all_y = list()
                for x, y in model_train_dataloader_fixed:
                    has_metadata = isinstance(x, (list, tuple))

                    x = x.to(DEVICE) if not has_metadata else [tensor.to(DEVICE) for tensor in x]
                    y = y.to(DEVICE)

                    em = nets[key]["model"].model(x, return_latent=True)

                    embedding.append(em.cpu())
                    all_y.append(y.cpu())

                embedding = torch.cat(embedding, dim=0).cpu()
                if "logit_norm" in key:
                    embedding = F.normalize(embedding, p=2, dim=1) / CONFIG["ood_detectors"]["logit_norm_temperature"]
                
                train_y = torch.cat(all_y, dim=0).cpu()

                # embedding = torch.randn((len(model_train_dataloader_fixed.dataset), 2), dtype=torch.get_default_dtype())
                # y = torch.tensor(np.random.choice([0, 1], size=embedding.size(0), replace=True), dtype=torch.long)
            nets[key]["train_embedding"] = embedding
            nets[key]["train_embedding_dataloader"] = torch.utils.data.DataLoader(
                MatrixDataset(nets[key]["train_embedding"], train_y),
                batch_size = 32,
                shuffle = False,
            )

            # Test embeddings
            with torch.no_grad():
                embedding = list()
                for x, y in model_test_dataloader:
                    has_metadata = isinstance(x, (tuple, list))

                    x = x.to(DEVICE) if not has_metadata else [tensor.to(DEVICE) for tensor in x]

                    em = nets[key]["model"].model(x, return_latent=True)
                    embedding.append(em.cpu())
                
                embedding = torch.cat(embedding, dim=0).cpu()
                if "logit_norm" in key:
                    embedding = F.normalize(embedding, p=2, dim=1) / CONFIG["ood_detectors"]["logit_norm_temperature"]
                
                # embedding = torch.randn((len(model_test_dataloader.dataset), 2), dtype=torch.get_default_dtype())

            nets[key]["test_embedding"] = embedding
            nets[key]["test_embedding_dataloader"] = torch.utils.data.DataLoader(
                MatrixDataset(nets[key]["test_embedding"], y=None),
                batch_size = 32,
                shuffle = False,
            )

            model_embedding_save_path = os.path.join(args.save, "models", key)
            torch.save(nets[key]["train_embedding"], os.path.join(model_embedding_save_path, "train_embedding.pt"))
            torch.save(train_y, os.path.join(model_embedding_save_path, "train_y.pt"))
            torch.save(nets[key]["test_embedding"], os.path.join(model_embedding_save_path, "test_embedding.pt"))

            nets[key]["model"].model.to("cpu")

    # KNN with all models and base
    for key in tqdm.tqdm(nets.keys(), "Train KNN"):
        detector_key = f"{key}_KNN"
        detectors[detector_key] = dict()
        detectors[detector_key]["ood_detector"] = ood_detectors.DeepNearestNeighbors(
            k = CONFIG["ood_detectors"]["knn_k"],
            threshold = CONFIG["ood_detectors"]["knn_threshold"],
            threshold_quantile = CONFIG["ood_detectors"]["knn_threshold_quantile"],
        )
        detectors[detector_key]["ood_detector"].train(
            nets[key]["train_embedding_dataloader"],
            device = DEVICE,
        )
        detectors[detector_key]["ood"] = detectors[detector_key]["ood_detector"].predict(
            nets[key]["test_embedding_dataloader"],
            device = DEVICE,
        )

    # Isolation Forest with all models and base
    for key in tqdm.tqdm(nets.keys(), "Train isolation forest"):
        detector_key = f"{key}_IF"
        detectors[detector_key] = dict()
        detectors[detector_key]["ood_detector"] = ood_detectors.IsolationForest()
        detectors[detector_key]["ood_detector"].train(
            nets[key]["train_embedding_dataloader"],
        )
        detectors[detector_key]["ood"] = detectors[detector_key]["ood_detector"].predict(
            nets[key]["test_embedding_dataloader"],
        )

    # Mahalanobis with all models
    for key in tqdm.tqdm(nets.keys(), "Train Mahalanobis"):
        detector_key = f"{key}_Mahalanobis"
        detectors[detector_key] = dict()
        detectors[detector_key]["ood_detector"] = ood_detectors.MahalanobisDistance(
            threshold = CONFIG["ood_detectors"]["mahalanobis_threshold"],
            threshold_quantile = CONFIG["ood_detectors"]["mahalanobis_threshold_quantile"]
        )
        detectors[detector_key]["ood_detector"].train(
            nets[key]["train_embedding_dataloader"],
        )
        detectors[detector_key]["ood"] = detectors[detector_key]["ood_detector"].predict(
            nets[key]["test_embedding_dataloader"],
        )

    # EBO with main model +/- logit norm
    for key in tqdm.tqdm(nets.keys(), "Train EBO"):
        detector_key = f"{key}_EBO"
        detectors[detector_key] = dict()
        detectors[detector_key]["ood_detector"] = ood_detectors.EBO(
            temperature = 1.0, # different temperature from logit norm
            threshold = CONFIG["ood_detectors"]["ebo_threshold"],
            threshold_quantile = CONFIG["ood_detectors"]["ebo_threshold_quantile"],
        )
        detectors[detector_key]["ood_detector"].train(
            nets[key]["train_embedding_dataloader"],
        )
        detectors[detector_key]["ood"] = detectors[detector_key]["ood_detector"].predict(
            nets[key]["test_embedding_dataloader"],
        )

    # MSP with main model +/- logit norm
    for key in tqdm.tqdm(nets.keys(), "Train MSP"):
        detector_key = f"{key}_MSP"
        detectors[detector_key] = dict()
        detectors[detector_key]["ood_detector"] = ood_detectors.MSP(
            method = "max",
            threshold = CONFIG["ood_detectors"]["msp_threshold"],
            threshold_quantile = CONFIG["ood_detectors"]["msp_threshold_quantile"],
        )
        detectors[detector_key]["ood_detector"].train(
            nets[key]["train_embedding_dataloader"],
        )
        detectors[detector_key]["ood"] = detectors[detector_key]["ood_detector"].predict(
            nets[key]["test_embedding_dataloader"],
        )

    # MSP entropy with main model +/- logit norm
    for key in tqdm.tqdm(nets.keys(), "Train MSP_entropy"):
        detector_key = f"{key}_MSP_entropy"
        detectors[detector_key] = dict()
        detectors[detector_key]["ood_detector"] = ood_detectors.MSP(
            method = "entropy",
            threshold = CONFIG["ood_detectors"]["msp_entropy_threshold"],
            threshold_quantile = CONFIG["ood_detectors"]["msp_entropy_threshold_quantile"],
        )
        detectors[detector_key]["ood_detector"].train(
            nets[key]["train_embedding_dataloader"],
        )
        detectors[detector_key]["ood"] = detectors[detector_key]["ood_detector"].predict(
            nets[key]["test_embedding_dataloader"],
        )
    

    # ---- Postprocess OOD detectors and models into ensembles ----
    # "...ensemble_4_2": This ensemble has 4 OOD detectors; this OOD detector is the 3rd of 4 (start indexing at 0)
    
    # Gather a list of unique OOD detectors (so all detectors in an ensemble are considered the same detector)
    unique_ood_detectors = list()
    for key in detectors.keys():
        if "ensemble" in key:
            try:
                ensemble_name = re.search(r"(ensemble_\d+)_\d+", key).group(1) # "ood_detector_ensemble_4_2" -> "ensemble_4"
                ensemble_id = re.search(r"(ensemble_\d+_\d+)", key).group(1) # "ood_detector_ensemble_4_2" -> "ensemble_4_2"
            except Exception as e:
                print(e)
                print(key)
                print(list(detectors.keys()))
                print(unique_ood_detectors)
                raise Exception

            unique_key = key.replace(ensemble_id, ensemble_name) # "ood_detector_ensemble_4_2" -> "ood_detector_ensemble_4"
            if unique_key in unique_ood_detectors:
                continue
            else:
                unique_ood_detectors.append(unique_key)
        else:
            unique_ood_detectors.append(key)
    
    print(f"Original OOD detector keys:\n{list(detectors.keys())}\nNew OOD detector keys:\n{unique_ood_detectors}")
    
    new_detectors = dict()
    for key in unique_ood_detectors:
        if "ensemble" in key:
            try:
                ensemble_name = re.search(r"(ensemble_\d+)", key).group(1) # "ood_detector_ensemble_4" -> "ensemble_4"
            except Exception as e:
                print(e)
                print(key)
                print(unique_ood_detectors)
                print(new_detectors.keys())
                raise Exception
            
            ensemble_size = int(ensemble_name.replace("ensemble_", ""))
            detector_keys = [key.replace(ensemble_name, f"{ensemble_name}_{i}") for i in range(ensemble_size)]
            detectors_in_ensemble = {detector_key: val for detector_key, val in detectors.items() if detector_key in detector_keys}
            
            print(f"Inferred {len(detectors_in_ensemble)} detectors belonging to {key}")

            new_detectors[key] = dict()
            new_detectors[key]["ood_detector"] = PretrainedOODEnsemble(
                ood_detectors = list(detectors_in_ensemble.keys()),
                predictions = [detectors_in_ensemble[k]["ood"] for k in detectors_in_ensemble.keys()],
                method = CONFIG["ood_detectors"]["ensemble_method"]
            )
            new_detectors[key]["ood"] = new_detectors[key]["ood_detector"].predict()

            print(f"Inferred {key} has {ensemble_size} detectors: {detector_keys}")
        else:
            new_detectors[key] = detectors[key]


    detectors = new_detectors

    print(f"Using these OOD detectors: {detectors.keys()}")

    # ---- Save intermediate model results ----
    model_intermediate_save_path = os.path.join(args.save, "models", "base") # should be created

    model = nets["base"]["model"]
    all_out, all_y = model.predict(model_train_dataloader_fixed, return_y=True, device=DEVICE)
    torch.save(all_out, os.path.join(model_intermediate_save_path, "train_predictions.pt"))
    # torch.save(all_y, os.path.join(model_intermediate_save_path, "train_y.pt")) # already saved earlier

    all_out, all_y = model.predict(model_val_dataloader, return_y=True, device=DEVICE)
    torch.save(all_out, os.path.join(model_intermediate_save_path, "val_predictions.pt"))
    torch.save(all_y, os.path.join(model_intermediate_save_path, "val_y.pt")) # already saved earlier

    all_out, all_y = model.predict(model_test_dataloader, return_y=True, device=DEVICE)
    torch.save(all_out, os.path.join(model_intermediate_save_path, "test_predictions.pt"))
    torch.save(all_y, os.path.join(model_intermediate_save_path, "test_y.pt"))

    # ---- Bootstrap results ----
    OOD_DETECTORS_SAVE = os.path.join(args.save, "ood_detectors")
    os.makedirs(OOD_DETECTORS_SAVE, exist_ok=True)

    for detector_name in detectors.keys():

        ood_detector_dir = os.path.join(OOD_DETECTORS_SAVE, detector_name)
        os.makedirs(ood_detector_dir, exist_ok=True)
    
        ood = detectors[detector_name]["ood"]
    
        ood_indices = np.argwhere(ood).flatten()
        id_indices = np.argwhere(1 - ood).flatten()

        ood_obs = dummy_test_dataset.subset(ood.astype(bool), inplace=False).obs
        id_obs = dummy_test_dataset.subset(~(ood.astype(bool)), inplace=False).obs

        # Save ID/OOD predictions
        id_obs.to_csv(os.path.join(ood_detector_dir, "id_obs.csv"))
        ood_obs.to_csv(os.path.join(ood_detector_dir, "ood_obs.csv"))

        with open(os.path.join(ood_detector_dir, "id_ood_indices.json"), "w") as f:
            f.write(json.dumps(
                {
                    "id_indices": [int(i) for i in id_indices],
                    "ood_indices": [int(i) for i in ood_indices],
                },
                indent = 2,
            ))

        # Set intersection
        print(set.intersection(set(ood_indices), set(id_indices))) # should be empty

        replicates = [str(i) for i in range(CONFIG["num_bootstrap_replicates"])] + ["main"]
        performance = {replicate: {"metadata": dict(), "ID": dict(), "OOD": dict(), "all": dict()} for replicate in replicates}

        for replicate in tqdm.tqdm(replicates, desc="Bootstrap"):

            # Subset to bootstrap subset
            subset_inds = np.random.choice(len(test_dataset), size=len(test_dataset), replace=True) if replicate != "main" else np.arange(len(test_dataset))

            # Compute ood, and then can re-define subsets based on that
            subset_ood_indices = [i for i in subset_inds if i in ood_indices] # index in the original dataset
            subset_id_indices = [i for i in subset_inds if i in id_indices]

            # Performance on OOD
            if 0 < len(subset_ood_indices):
                out = all_out[subset_ood_indices]
                y = all_y[subset_ood_indices]

                performance[replicate]["OOD"]["accuracy"] = float((np.argmax(out, axis=1) == y).float().mean())
                out_probs = out if torch.all(torch.abs(torch.sum(out, dim=1) - 1.0) < 1e-4) else nn.Softmax(dim=1)(out)
                performance[replicate]["OOD"]["AUROC"] = sklearn.metrics.roc_auc_score(
                    y_true = nn.functional.one_hot(y, num_classes=len(np.unique(all_y))).detach().cpu().numpy(),
                    y_score = out_probs.detach().cpu().numpy()
                ) if len(np.unique(y)) > 1 else float("nan")
            else:
                performance[replicate]["OOD"]["accuracy"] = float("nan")
                performance[replicate]["OOD"]["AUROC"] = float("nan")

            # Performance on ID
            if 0 < len(subset_id_indices):
                out = all_out[subset_id_indices]
                y = all_y[subset_id_indices]

                performance[replicate]["ID"]["accuracy"] = float((np.argmax(out, axis=1) == y).float().mean())
                out_probs = out if torch.all(torch.abs(torch.sum(out, dim=1) - 1.0) < 1e-4) else nn.Softmax(dim=1)(out)
                performance[replicate]["ID"]["AUROC"] = sklearn.metrics.roc_auc_score(
                    y_true = nn.functional.one_hot(y, num_classes = len(np.unique(all_y))).detach().cpu().numpy(),
                    y_score = out_probs.detach().cpu().numpy()
                ) if len(np.unique(y)) > 1 else float("nan")
            else:
                performance[replicate]["ID"]["accuracy"] = float("nan")
                performance[replicate]["ID"]["AUROC"] = float("nan")
            
            # Performance on all
            if 0 < len(subset_inds):
                out = all_out[subset_inds]
                y = all_y[subset_inds]

                performance[replicate]["all"]["accuracy"] = float((np.argmax(out, axis=1) == y).float().mean())
                out_probs = out if torch.all(torch.abs(torch.sum(out, dim=1) - 1.0) < 1e-4) else nn.Softmax(dim=1)(out)
                performance[replicate]["all"]["AUROC"] = sklearn.metrics.roc_auc_score(
                    y_true = nn.functional.one_hot(y, num_classes = len(np.unique(all_y))).detach().cpu().numpy(),
                    y_score = out_probs.detach().cpu().numpy()
                ) if len(np.unique(y)) > 1 else float("nan")
            else:
                performance[replicate]["all"]["accuracy"] = float("nan")
                performance[replicate]["all"]["AUROC"] = float("nan")

            performance[replicate]["metadata"] = {
                "num_ood": len(subset_ood_indices),
                "num_id": len(subset_id_indices),
            }

        with open(os.path.join(ood_detector_dir, "performance.json"), "w") as f:
            f.write(json.dumps(performance, indent=2))

    STDOUT_WRITE_FILE.close()
    STDERR_WRITE_FILE.close()