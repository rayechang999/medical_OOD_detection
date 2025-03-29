import torch
import torch.nn as nn
import torch.nn.functional as F

class LogitNormLoss(nn.Module):
    def __init__(self, temperature: float, *crossentropy_args, **crossentropy_kwargs):
        """
        Logit normalization loss from Wei et al. in ICML 2022. Adapted from the original
        code for logit normalization at github.com/hongxin001/logitnorm_ood; please
        reference Wei et al.'s repo for the original implementation.

        Parameters
        ----------
        temperature : float
            Temperature for norm scaling. In the original paper (Wei et al. in ICML 2022),
            the authors set the temperature to 0.04 for CIFAR-10.
        """
        super(LogitNormLoss, self).__init__()

        self.temperature = temperature
        self.crossentropy = nn.CrossEntropyLoss(*crossentropy_args, **crossentropy_kwargs)
    
    def forward(self, out: torch.Tensor, y: torch.Tensor):
        out = F.normalize(out, p=2, dim=1) / self.temperature

        return self.crossentropy(out, y)

def get_loss_fn(loss: str, loss_args=None, loss_kwargs=None):
    """
    Returns the torch loss function given an input string.
    """
    if loss_args is None:
        loss_args = list()
    if loss_kwargs is None:
        loss_kwargs = dict()
    
    if loss == "BCELoss":
        return nn.BCELoss(*loss_args, **loss_kwargs)
    elif loss == "CrossEntropyLoss":
        return nn.CrossEntropyLoss(*loss_args, **loss_kwargs)
    elif loss == "LogitNormLoss":
        return LogitNormLoss(*loss_args, **loss_kwargs)
    else:
        raise ValueError(f"Loss type '{loss}' is not supported")