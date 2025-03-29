import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import sklearn
import numpy as np

import tqdm

from .base import BaseModel
from . import utils

class M0(nn.Module):
    def __init__(self, input_shape, out_features):
        """
        Helper class for Conv2D. This class defines a nn.Module, the actual 
        neural network which Conv2D wraps around.
        """
        super(M0, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(self.compute_fc_in_features(input_shape), 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, out_features)
        self.dropout = nn.Dropout(0.5)
    
    def compute_fc_in_features(self, input_shape):
        x = torch.zeros(1, *input_shape, device=next(self.parameters()).device)

        with torch.no_grad():
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)

        return x.numel()
    
    def forward(self, x):
        x = x.to(self.device)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

    @property
    def device(self):
        return next(self.conv1.parameters()).device

class M1(nn.Module):
    def __init__(self, input_shape, out_features):
        """
        Helper class for Conv2D. This class defines a nn.Module, the actual 
        neural network which Conv2D wraps around.
        """
        super(M1, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(self.compute_fc_in_features(input_shape), 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, out_features)
        self.dropout = nn.Dropout(0.5)
    
    def compute_fc_in_features(self, input_shape):
        x = torch.zeros(1, *input_shape, device=next(self.parameters()).device)

        with torch.no_grad():
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = self.pool(x)

        return x.numel()
    
    def forward(self, x):
        x = x.to(self.device)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

    @property
    def device(self):
        return next(self.conv1.parameters()).device

class M2(nn.Module):
    def __init__(self, input_shape, out_features):
        """
        Helper class for Conv2D. This class defines a nn.Module, the actual 
        neural network which Conv2D wraps around.
        """
        super(M2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(self.compute_fc_in_features(input_shape), 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, out_features)
        self.dropout = nn.Dropout(0.5)
    
    def compute_fc_in_features(self, input_shape):
        x = torch.zeros(1, *input_shape, device=next(self.parameters()).device)

        with torch.no_grad():
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = self.pool(x)
            x = F.relu(self.conv5(x))
            x = F.relu(self.conv6(x))
            x = self.pool(x)

        return x.numel()
    
    def forward(self, x):
        x = x.to(self.device)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

    @property
    def device(self):
        return next(self.conv1.parameters()).device

class M3(nn.Module):
    def __init__(self, input_shape, out_features):
        super(M3, self).__init__()
        self.input_shape = input_shape
        self.out_features = out_features

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.compute_fc_in_features(input_shape), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.output = nn.Linear(64, out_features)

    def compute_fc_in_features(self, input_shape):
        x = torch.zeros(1, *input_shape, device=next(self.parameters()).device)

        with torch.no_grad():
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)

        return x.numel()
    
    def forward(self, x):
        x = x.to(self.device)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x)
        return x

    @property
    def device(self):
        return next(self.block1.parameters()).device

class M3WithMetadata(nn.Module):
    def __init__(self, input_shape, out_features, metadata_in_features, metadata_out_features=128):
        """
        The input_shape should *not* contain the batch dimension.
        """
        super(M3WithMetadata, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.compute_fc_in_features(input_shape) + metadata_out_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_features),
        )

        self.metadata_model = nn.Sequential(
            nn.Linear(metadata_in_features, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(512, metadata_out_features),
            nn.BatchNorm1d(metadata_out_features),
            nn.ReLU(),
        )

    def compute_fc_in_features(self, input_shape):
        x = torch.zeros(1, *input_shape, device=next(self.parameters()).device)

        with torch.no_grad():
            x = self.conv(x)

        return x.numel()
    
    def forward(self, x):
        x, metadata = x
        x, metadata = x.to(self.device), metadata.to(self.device)

        x = self.conv(x)
        x = x.view(x.size(0), -1)

        metadata = self.metadata_model(metadata)

        x = torch.cat([x, metadata], dim=-1)
        x = self.fc(x)
        return x
    
    def forward(self, x):
        x, metadata = x
        x, metadata = x.to(self.device), metadata.to(self.device)

        x = self.conv(x)
        x = x.view(x.size(0), -1)

        metadata = self.metadata_model(metadata)

        x = torch.cat([x, metadata], dim=-1)
        x = self.fc(x)

        return x

    @property
    def device(self):
        return next(self.metadata_model.parameters()).device
    

class GuanVersion1_4000(nn.Module):
    def __init__(self, input_shape, out_features):
        """
        Adapted version of the model at https://github.com/GuanLab/PDDB/blob/master/model/guan_version1_4000.py
        """
        super(GuanVersion1_4000, self).__init__()
        
        # Input dimensions
        self.input_shape = input_shape
        self.out_features = out_features
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_shape[0], out_channels=8, kernel_size=5),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=4),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2),
        )

        # Fully connected layer
        self.fc = nn.Linear(self.calculate_output_length(input_shape), out_features)

    def forward(self, x, return_latent=False):

        x = self.conv(x)
        x = x.view(x.size(0), -1)

        if return_latent:
            return x
        
        x = self.fc(x)
        return x
    
    def calculate_output_length(self, input_shape):
        x = torch.zeros(input_shape).unsqueeze(0)

        with torch.no_grad():
            x = self.conv(x)
        
        return x.numel()

class Conv1DJosh(nn.Module):
    """
    Written by Josh; adapted from data/projects/.../external/code_mix_pytorch/conv.py
    """
    def __init__(self, input_size, channels):
        super(Conv1DJosh, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, 8, kernel_size=5, padding='same'),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(8, 16, kernel_size=5, padding='same'),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=4, padding='same'),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 32, kernel_size=4, padding='same'),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding='same'),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 64, kernel_size=4, padding='same'),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding='same'),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)  # Binary classification
        )
        
    def forward(self, x, return_latent=False):
        features = self.conv(x)

        if return_latent:
            latent = nn.AdaptiveAvgPool1d(1)(features)
            latent = latent.view(latent.size(0), -1)
            return latent
        
        return self.classifier(features)

    def extract_features(self, x):
        return self.conv(x)


class SimpleFCTranscriptomics(nn.Module):
    def __init__(self, input_shape: tuple):
        """
        input_shape must exclude the batch dimension
        """
        super(SimpleFCTranscriptomics, self).__init__()

        self.net1 = nn.Sequential(
            nn.Linear(input_shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )
        self.net2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32, 2),
        )
    
    def forward(self, x, return_latent=False):
        x = self.net1(x)

        if return_latent:
            return x
        
        x = self.net2(x)
        return x


class Conv2D(BaseModel):
    def __init__(
            self, 
            input_shape=None, 
            out_features: int=None, 
            loss: str=None, 
            loss_args=None, 
            loss_kwargs=None, 
            model: str="M0", 
            metadata_in_features=None, 
            metadata_out_features=128,
            logit_norm=False,
        ):
        """
        Parameters
        ----------
        logit_norm : bool, default: False
            Logit normalization (Wei et al. in ICML 2022). Whether or not to normalize
            the logits of the model before outputting them. This can improve OOD detection
            performance.
        """
        if model == "M0":
            self.model = M0(input_shape, out_features, logit_norm=logit_norm)
        elif model == "M1":
            self.model = M1(input_shape, out_features)
        elif model == "M2":
            self.model = M2(input_shape, out_features)
        elif model == "M3":
            self.model = M3(input_shape, out_features)
        elif model == "M3WithMetadata":
            self.model = M3WithMetadata(input_shape, out_features, metadata_in_features)
        elif model == "GuanVersion1_4000":
            self.model = GuanVersion1_4000(input_shape=input_shape, out_features=out_features)
        elif model == "Conv1DJosh":
            self.model = Conv1DJosh(input_size=None, channels=channels)
        elif model == "SimpleFCTranscriptomics":
            self.model = SimpleFCTranscriptomics(input_shape=input_shape)
        else:
            raise Exception
        
        self.out_features = out_features

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = utils.get_loss_fn(loss, loss_args, loss_kwargs)

        if logit_norm and loss != "LogitNormLoss":
            print(f"Warning: You are using logit normalization but aren't using LogitNormLoss. Are you sure this is what you want?")

    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_epochs: int,
        device,
        val_dataloader=None,
        save: str=None,
        save_freq: int=None,
    ):
        self.model.to(device)

        metrics = {
            "train_loss": list(),
            "train_accuracy": list(),
            "train_AUROC": list(),
            "val_loss": list(),
            "val_accuracy": list(),
            "val_AUROC": list(),
        }

        for epoch in tqdm.tqdm(range(num_epochs), desc="Training model Conv2DM0"):

            self.model.train()

            epoch_loss = 0
            epoch_accuracy = 0
            all_outs = list()
            all_y = list()
            num_observations = 0

            for x, y in tqdm.tqdm(dataloader, desc=f"Model training (epoch {epoch})"):

                has_metadata = isinstance(x, (tuple, list))

                x = x.to(device) if not has_metadata else [tensor.to(device) for tensor in x]
                y = y.to(device)

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

            epoch_loss /= num_observations
            epoch_accuracy /= num_observations

            all_outs = torch.cat(all_outs, dim=0)
            all_y = torch.cat(all_y).squeeze()

            out_probs = all_outs if torch.all(torch.abs(torch.sum(all_outs, dim=1) - 1.0) < 1e-4) else nn.Softmax(dim=1)(all_outs)
            epoch_AUROC = sklearn.metrics.roc_auc_score(
                y_true = nn.functional.one_hot(all_y, num_classes=self.out_features).detach().cpu().numpy(),
                y_score = out_probs.detach().cpu().numpy()
            )

            print(epoch, epoch_loss, epoch_accuracy, epoch_AUROC)

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

                        has_metadata = isinstance(x, (tuple, list))

                        x = x.to(device) if not has_metadata else [tensor.to(device) for tensor in x]
                        y = y.to(device)

                        out = self.model(x)
                        loss = self.loss_fn(out, y)

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
                        y_true = nn.functional.one_hot(all_y, num_classes=self.out_features).detach().cpu().numpy(),
                        y_score = out_probs
                    )

                print(epoch, val_loss, val_accuracy, val_AUROC)
                
                metrics["val_loss"].append(float(val_loss))
                metrics["val_accuracy"].append(float(val_accuracy))
                metrics["val_AUROC"].append(float(val_AUROC))
            

            if save is not None and save_freq is not None and epoch % save_freq == 0:
                torch.save(self.model.state_dict(), os.path.join(save, f"epoch_{epoch}.pt"))
            
        if save is not None:
            torch.save(self.model.state_dict(), os.path.join(save, "final.pt"))
        
        return metrics
    
    def predict(self, dataloader, return_y=False, device=None):
        if isinstance(dataloader.sampler, torch.utils.data.sampler.RandomSampler):
            raise Exception(f"The dataloader must load observations in a fixed order. Try initializing with shuffle=False.")
        
        self.model.eval()
        if device is not None:
            self.model.to(device)
        
        outs = list()
        ys = list()
        with torch.no_grad():
            for x, y in dataloader:

                has_metadata = isinstance(x, (list, tuple))

                if device is not None:
                    x = x.to(device) if not has_metadata else [tensor.to(device) for tensor in x]
                    y = y.to(device)

                out = self.model(x).detach().cpu()
                outs.append(out)

                if return_y:
                    ys.append(y.detach().cpu())
        
        out = torch.cat(outs, dim=0)
        if return_y:
            y = torch.cat(ys)
        
        return (out, y) if return_y else out