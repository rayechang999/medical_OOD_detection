"""
Implementation of the algorithm described in "Out of Distribution Detection with 
Deep Nearest Neighbors" Sun et al. ICML 2022.

A deep neural network classifier is trained with contrastive learning, which 
makes each individual training example its own class in order to learn a more 
rich latent space. The penultimate activations of this neural network are used as 
the latent embeddings of the training examples. At prediction time, the prediction 
sample is also embedded, and the distance between the new embedding and the k-th 
nearest training embedding is computed. If this distance is above a threshold, 
then the sample is labeled as OOD.
"""
import faiss
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import OODDetector

import tqdm

class FullyConnected(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        """
        Simple feedforward neural network to be used with DeepNearestNeighbors.
        """
        super(FullyConnected, self).__init__()

        self.L1 = nn.Linear(in_features, 128)
        self.a1 = nn.ReLU()
        self.L2 = nn.Linear(128, 64)
        self.a2 = nn.ReLU()
        self.L3 = nn.Linear(64, 32)
        self.a3 = nn.ReLU()
        self.LP = nn.Linear(32, 16)
        self.aP = nn.ReLU()
        self.LF = nn.Linear(16, out_features)

    
    def forward(self, x, return_latent: bool=False, return_penultimate_postact: bool=False):
        z = self.L1(x)
        z = self.a1(z)
        z = self.L2(z)
        z = self.a2(z)
        z = self.L3(z)
        z = self.a3(z)
        z = self.LP(z)
        if return_latent: return z
        z = self.aP(z)
        if return_penultimate_postact: return z
        z = self.LF(z)
        return z

class Conv2DM0(nn.Module):
    def __init__(self, input_shape, out_features):
        """
        Helper class for Conv2D. This class defines a nn.Module, the actual 
        neural network which Conv2D wraps around.
        """
        super(Conv2DM0, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(self.compute_fc_in_features(input_shape), 128)
        self.fc2 = nn.Linear(128, out_features)
        # self.dropout = nn.Dropout(0.5)
    
    def compute_fc_in_features(self, input_shape):
        x = torch.zeros(1, *input_shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        return x.numel()
    
    def forward(self, x, return_latent=False, return_penultimate_postact=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        if return_latent: return x
        x = F.relu(x)
        if return_penultimate_postact: return x
        # x = self.dropout(x)
        x = self.fc2(x)

        return x

class DeepNearestNeighbors(OODDetector):
    def __init__(
            self, 
            k: int,
            threshold: float=None,
            threshold_quantile: float=None,
        ):  
        if sum([threshold is None, threshold_quantile is None]) != 1:
            raise ValueError(f"Excatly one of threshold or threshold_quantile must be specified")
        
        self.k = k
        self.threshold = threshold
        self.threshold_quantile = threshold_quantile

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        self.data = {
            "epochs": [],
            "losses": [],
        }
    
    def train(self, dataloader, device=None):
        metrics = dict()
        self.embed(dataloader, device)
        
        if self.threshold is None:
            print(f"Computing quantile threshold...")

            dist = list()
            for x, y in dataloader:

                if isinstance(x, torch.Tensor):
                    dist += [self._predict_single(xi.unsqueeze(0)) for xi in x]
                elif isinstance(x, (list, tuple)):
                    dist += [self._predict_single([elt[i].unsqueeze(0) for elt in x]) for i in range(len(x[0]))]
                else:
                    raise Exception

            dist = np.array(dist)

            self.threshold = float(np.quantile(dist, self.threshold_quantile))
            print(f"Computed {self.threshold} as the threshold: {round(100 * (dist > self.threshold).mean(), 2)}% of training examples labeled as OOD")

        metrics["threshold"] = self.threshold
        metrics["threshold_quantile"] = self.threshold_quantile

        return metrics
    
    def embed(self, dataloader, device=None):
        """
        Stores the embeddings to be used later in predict. The x should be the
        training data.
        """
        train_embeddings = list()

        with torch.no_grad():
            for x, y in dataloader:
                if isinstance(x, (tuple, list)):
                    x, metadata = x
                
                embed = self.flatten(x).cpu()
                train_embeddings.append(embed)
            
        train_embeddings = torch.cat(train_embeddings, axis=0).detach().cpu().numpy()

        self.embedding = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
        self.index = faiss.IndexFlatL2(np.shape(self.embedding)[1])
        self.index.add(self.embedding)
        
    def _predict_single(self, x):
        """
        Assumes x is shape (1, ...) (i.e. it has a batch dimension of 1)
        """
        with torch.no_grad():
            z = self.flatten(x).cpu().numpy()
            z = z / np.linalg.norm(z, axis=1)
        
        try:
            D, I = self.index.search(z, self.k)
        except ValueError:
            print(np.shape(z))
            print(np.expand_dims(z, axis=0).shape)
            print(z)
            print(self.k)
            raise ValueError

        norm = np.linalg.norm(self.embedding[I[0, self.k - 1], :] - np.squeeze(z))
        norm_2 = np.sqrt(D[0, self.k - 1])

        if not (np.abs(norm - norm_2) < 0.001):
            print(norm)
            print(norm_2)
            print(D)
            print(I)
            print(z)
            raise Exception

        return norm
    
    def predict(self, dataloader, return_score=False, device=None) -> np.ndarray:

        if isinstance(dataloader.sampler, torch.utils.data.sampler.RandomSampler):
            raise Exception(f"The dataloader must load observations in a fixed order. Try initializing with shuffle=False.")

        outs = list()
        
        for x, y in dataloader:
            
            if isinstance(x, torch.Tensor):
                for xi in x:
                    out = self._predict_single(xi.unsqueeze(0))
                    outs.append(out)
            elif isinstance(x, (list, tuple)):
                for i in range(len(x[0])):
                    out = self._predict_single([elt[i].unsqueeze(0) for elt in x])
                    outs.append(out)
            else:
                raise Exception

        return np.array(outs) if return_score else (np.array(outs) > self.threshold).astype(int)



class DeepNearestNeighborsWithModel(OODDetector):
    def __init__(
        self, 
        k: int,
        threshold: float=None,
        threshold_quantile: float=None,
        model: nn.Module=None
    ):
        self.model = model
        
        if sum([threshold is None, threshold_quantile is None]) != 1:
            raise ValueError(f"Excatly one of threshold or threshold_quantile must be specified")
        
        self.k = k
        self.threshold = threshold
        self.threshold_quantile = threshold_quantile

        self.data = {
            "epochs": [],
            "losses": [],
        }
    
    def train(self, dataloader, device=None):
        if device is not None:
            self.model.to(device)

        metrics = {
            
        }
        self.embed(dataloader, device)
        
        if self.threshold is None:
            print(f"Computing quantile threshold...")

            dist = list()
            for x, y in dataloader:

                if isinstance(x, torch.Tensor):
                    dist += [self._predict_single(xi.unsqueeze(0)) for xi in x]
                elif isinstance(x, (list, tuple)):
                    dist += [self._predict_single([elt[i].unsqueeze(0) for elt in x]) for i in range(len(x[0]))]
                else:
                    raise Exception

            dist = np.array(dist)

            self.threshold = float(np.quantile(dist, self.threshold_quantile))
            print(f"Computed {self.threshold} as the threshold: {round(100 * (dist > self.threshold).mean(), 2)}% of training examples labeled as OOD")

        metrics["threshold"] = self.threshold
        metrics["threshold_quantile"] = self.threshold_quantile

        return metrics
    
    def embed(self, dataloader, device=None):
        """
        Stores the embeddings to be used later in predict. The x should be the
        training data.
        """
        self.model.eval()
        if device is not None:
            self.model.to(device)

        train_embeddings = list()

        with torch.no_grad():
            for x, y in dataloader:

                embed = self.model(x, return_latent=True).cpu()
                # train_embeddings = [train_embeddings[i, :] for i in range(np.shape(train_embeddings)[0])]
                # train_embeddings = [e / np.linalg.norm(e) for e in train_embeddings]
                # self.embedding = train_embeddings
                train_embeddings.append(embed)
            
        train_embeddings = torch.cat(train_embeddings, axis=0).detach().cpu().numpy()

        self.embedding = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
        
    def _predict_single(self, x):
        self.model.eval()

        with torch.no_grad():
            z = self.model(x, return_latent=True).detach().cpu().numpy()
            
            z = z / np.linalg.norm(z, axis=1)
        
        norms = np.linalg.norm(self.embedding - z, axis=1).squeeze()
        norms = np.sort(norms)

        return norms[self.k]
    
    def predict(self, dataloader, return_score=False, device=None) -> np.ndarray:

        if isinstance(dataloader.sampler, torch.utils.data.sampler.RandomSampler):
            raise Exception(f"The dataloader must load observations in a fixed order. Try initializing with shuffle=False.")

        if device is not None:
            self.model.to(device)

        outs = list()
        
        for x, y in dataloader:
            
            if isinstance(x, torch.Tensor):
                for xi in x:
                    out = self._predict_single(xi.unsqueeze(0))
                    outs.append(out)
            elif isinstance(x, (list, tuple)):
                for i in range(len(x[0])):
                    out = self._predict_single([elt[i].unsqueeze(0) for elt in x])
                    outs.append(out)
            else:
                raise Exception

        return np.array(outs) if return_score else (np.array(outs) > self.threshold).astype(int)
