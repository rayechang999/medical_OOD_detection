import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import sklearn
import numpy as np

import tqdm

from .base import BaseModel
from . import utils

import timm

class M0(nn.Module):
    def __init__(self, in_features, out_features):
        super(M0, self).__init__()
        
        self.fc = nn.Linear(in_features, out_features)
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
    
    def forward(self, x):
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.fc(dropout(x))
            else:
                out += self.fc(dropout(x))
        
        out /= len(self.dropouts)

        return out

class M1(nn.Module):
    def __init__(self, in_features, out_features):
        super(M1, self).__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.fc2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32, out_features)
        )
    
    def forward(self, x, return_latent=False):
        x = self.fc1(x)

        if return_latent:
            return x
        
        x = self.fc2(x)

        return x

class EfficientNetNN(nn.Module):
    def __init__(self, model, out_features, efficientnet="B3", fine_tune=False):
        super(EfficientNetNN, self).__init__()

        if model not in ("M0", "M1"):
            raise ValueError(f"Model {model} not supported")
        if efficientnet not in ("B3",):
            raise ValueError(f"Currently, model {efficientnet} is not supported.")
        
        self.effnet = timm.create_model(f"efficientnet_{efficientnet.lower()}", pretrained=True)
        self.fine_tune = fine_tune
        if not self.fine_tune:
            for param in self.effnet.parameters():
                param.requires_grad = False
        
        if model == "M0":
            self.model = M0(in_features=1000, out_features=out_features)
        elif model == "M1":
            self.model = M1(in_features=1000, out_features=out_features)
    
    def forward(self, x):
        x = x.to(self.device)

        if self.fine_tune:
            x = self.effnet(x)
        else:
            with torch.no_grad():
                x = self.effnet(x)
        
        x = self.model(x)

        return x
    
    def train(self, mode=True):

        super(EfficientNetNN, self).train(mode)

        if not self.fine_tune:
            self.effnet.eval()
        
        return self
    
    def eval(self):
        return self.train(False)
    
    @property
    def device(self):
        return next(self.model.parameters()).device

class EfficientNetNNWithMetadata(nn.Module):
    def __init__(self, model, out_features, metadata_in_features: int, metadata_out_features=128, efficientnet="B3", fine_tune=False):
        super(EfficientNetNNWithMetadata, self).__init__()

        if model not in ("M0", "M1"):
            raise ValueError(f"Model {model} not supported")
        if efficientnet not in ("B3",):
            raise ValueError(f"Currently, model {efficientnet} is not supported.")
        
        self.effnet = timm.create_model(f"efficientnet_{efficientnet.lower()}", pretrained=True)
        self.fine_tune = fine_tune
        if not self.fine_tune:
            for param in self.effnet.parameters():
                param.requires_grad = False
        
        if model == "M0":
            self.model = M0(in_features=1000+metadata_out_features, out_features=out_features)
        elif model == "M1":
            self.model = M1(in_features=1000+metadata_out_features, out_features=out_features)

        self.metadata_model = nn.Sequential(
            nn.Linear(metadata_in_features, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(512, metadata_out_features),
            nn.BatchNorm1d(metadata_out_features),
            nn.ReLU(),
        )
    
    def forward(self, x, return_latent=False):
        x, metadata = x # metadata is a dict
        x, metadata = x.to(self.device), metadata.to(self.device)

        if self.fine_tune:
            x = self.effnet(x)
        else:
            with torch.no_grad():
                x = self.effnet(x)
        
        metadata = self.metadata_model(metadata)

        x = torch.cat([x, metadata], dim=-1) # x and metadata should both be 1d by this point
        
        x = self.model(x, return_latent=return_latent)
        
        return x
    
    def train(self, mode=True):

        super(EfficientNetNNWithMetadata, self).train(mode)

        if not self.fine_tune:
            self.effnet.eval()
        
        return self
    
    def eval(self):
        return self.train(False)
    
    @property
    def device(self):
        return next(self.metadata_model.parameters()).device

class EfficientNet(BaseModel):
    def __init__(self, model, out_features, loss, loss_args=None, loss_kwargs=None, efficientnet="B3", fine_tune=False, metadata_in_features=None):
        if metadata_in_features is not None:
            self.model = EfficientNetNNWithMetadata(model=model, metadata_in_features=metadata_in_features, out_features=out_features, efficientnet=efficientnet, fine_tune=fine_tune)
        else:
            self.model = EfficientNetNN(model=model, out_features=out_features, efficientnet=efficientnet, fine_tune=fine_tune)
        
        self.loss_fn = utils.get_loss_fn(loss)
        self.optimizer = torch.optim.Adam(
            [
                {"params": filter(lambda p: p.requires_grad, self.model.effnet.parameters()), "lr": 1e-5},
                {"params": self.model.model.parameters(), "lr": 1e-3}
            ]
        )
        self.out_features = out_features
        
    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_epochs: int,
        device=None,
        val_dataloader=None,
        save: str=None,
        save_freq: int=None,
    ):
        if device is not None:
            self.model.to(device)

        metrics = {
            "train_loss": list(),
            "train_accuracy": list(),
            "train_AUROC": list(),
            "val_loss": list(),
            "val_accuracy": list(),
            "val_AUROC": list(),
        }

        for epoch in tqdm.tqdm(range(num_epochs), desc="Training model EfficientNet"):

            self.model.train()

            epoch_loss = 0
            epoch_accuracy = 0
            all_outs = list()
            all_y = list()
            num_observations = 0

            batch_num = 0

            for x, y in tqdm.tqdm(dataloader, desc=f"Model training (epoch {epoch})"):

                self.optimizer.zero_grad()

                out = self.model(x)
                loss = self.loss_fn(out, y)

                loss.backward()
                self.optimizer.step()

                n = x.size(0) if isinstance(x, torch.Tensor) else x[0].size(0)

                epoch_loss += loss.detach().cpu().item() * n
                epoch_accuracy += float((torch.argmax(out, dim=1) == y).detach().cpu().float().mean() * n)
                
                all_outs.append(out.detach().cpu())
                all_y.append(y.detach().cpu())
                num_observations += float(n)

                batch_num += 1

            epoch_accuracy /= num_observations

            all_outs = torch.cat(all_outs, dim=0)
            all_y = torch.cat(all_y).squeeze()

            out_probs = all_outs if torch.all(torch.abs(torch.sum(all_outs, dim=1) - 1.0) < 1e-4) else nn.Softmax(dim=1)(all_outs)
            epoch_AUROC = sklearn.metrics.roc_auc_score(
                y_true = nn.functional.one_hot(all_y, num_classes=self.out_features).detach().cpu().numpy(),
                y_score = out_probs.detach().cpu().numpy()
            )

            print(f"\nEpoch {epoch} | train_loss {epoch_loss} | train_accuracy {epoch_accuracy} | train_AUROC {epoch_AUROC}")

            metrics["train_loss"].append(float(epoch_loss))
            metrics["train_accuracy"].append(float(epoch_accuracy))
            metrics["train_AUROC"].append(float(epoch_AUROC))

            if val_dataloader is not None:

                self.model.eval()

                with torch.no_grad():
                    
                    val_loss = 0
                    val_accuracy = 0
                    all_outs = list()
                    all_y = list()
                    num_observations = 0

                    for x, y in val_dataloader:

                        out = self.model(x)
                        loss = self.loss_fn(out, y)

                        n = x.size(0) if isinstance(x, torch.Tensor) else x[0].size(0)

                        val_loss += loss.detach().cpu().item() * n
                        val_accuracy += float((torch.argmax(out, dim=1) == y).detach().cpu().float().mean() * n)
                        
                        all_outs.append(out.detach().cpu())
                        all_y.append(y.detach().cpu())
                        num_observations += float(n)

                    val_accuracy /= num_observations

                    all_outs = torch.cat(all_outs, dim=0)
                    all_y = torch.cat(all_y).squeeze()

                    out_probs = all_outs if torch.all(torch.abs(torch.sum(all_outs, dim=1) - 1.0) < 1e-4) else nn.Softmax(dim=1)(all_outs)
                    val_AUROC = sklearn.metrics.roc_auc_score(
                        y_true = nn.functional.one_hot(all_y, num_classes=self.out_features).detach().cpu().numpy(),
                        y_score = out_probs
                    )
                
                print(f"\nEpoch {epoch} | val_loss {val_loss} | val_accuracy {val_accuracy} | val_AUROC {val_AUROC}")

                metrics["val_loss"].append(float(val_loss))
                metrics["val_accuracy"].append(float(val_accuracy))
                metrics["val_AUROC"].append(float(val_AUROC))
            

            if save is not None and save_freq is not None and epoch % save_freq == 0:
                torch.save(self.model.state_dict(), os.path.join(save, f"epoch_{epoch}.pt"))
        
        if save is not None:
            torch.save(self.model.state_dict(), os.path.join(save, "final.pt"))
        
        return metrics
    
    def predict(self, dataloader, return_y=False, device=None):

        self.model.eval()

        if device is not None:
            self.model.to(device)
        
        outs = list()
        ys = list()
        with torch.no_grad():
            for x, y in dataloader:
                
                out = self.model(x).detach().cpu()
                outs.append(out)

                if return_y:
                    ys.append(y.detach().cpu())
        
        out = torch.cat(outs, dim=0)
        if return_y:
            y = torch.cat(ys)
        
        return (out, y) if return_y else out