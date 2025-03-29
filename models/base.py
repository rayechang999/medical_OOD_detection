# Base model from which all others inherit

class BaseModel:
    def __init__(self, *args, **kwargs):
        """
        Initializes the model. Usually, one of the arguments will be a string 
        `model`, which can be used to specify the exact architecture of this
        class of model. For example, when initializing `Conv2D(...)`, doing
        `Conv2D(model="M2")` will use the "M2" model (see models/conv2d.py for 
        the exact details of this architecture). Other common parameters include 
        the loss function type and learning rate.

        If you need access to the actual model (an object of type nn.Module), 
        it's usually stored in self.model.
        """
        pass
    
    def train(self, dataloader, *args, **kwargs) -> dict:
        """
        Trains the model on `(x, y)` pairs returned from a particular `dataloader`.

        Args:
            dataloader (torch.utils.data.DataLoader): A dataloader that returns
                (x, y) pairs.
        
        Returns:
            A dictionary of training metrics.
        """
        pass

    def predict(self, dataloader, return_y: bool=False, return_latent: bool=False, *args, **kwargs):
        """
        Predicts output using a given `dataloader` as input. This class assumes 
        that all of the raw data are too big to fit into memory at once, which 
        is why it uses a dataloader instead.

        Args:
            dataloader (torch.utils.data.DataLoader): A dataloader that returns 
                (x, y) pairs. Only x will be used as input to the model.
            return_y (bool): Whether or not to return a torch.Tensor that 
                contains all the `y` from the dataloader, concatenated in the 
                order that they were read. This can be useful for computing 
                downstream metrics (like AUROC), without having to load in 
                the entire dataset at once.
            return_latent (bool): Whether to return the neural network's intermediate
                layer representation of the data (return_latent == True) or return 
                the actual output of the neural network (return_latent == False).
                Returning intermediate/latent representation is sometimes useful
                for OOD detection algorithms.
        
        Returns:
            out: torch.Tensor or a tuple (out: torch.Tensor, y: torch.Tensor).
            If return_y is False, then just returns the neural network outputs 
            `out`, otherwise returns both the `out` and the `y`. If return_latent
            is True, then `out` represents the intermediate activations of the 
            neural network rather than the final output.
        """
        pass