# Base OOD detector interface, shared across specific detection algorithms

import numpy as np

class OODDetector:
    def __init__(self, *args, **kwargs):
        """
        Initializes the OOD detection algorithm. Common arguments include 
        the specific model architecture to be used (for DeepNearestNeighbors),
        or an nn.Module whose intermediate layer is to be extracted and used
        (for DeepNearestNeighborsWithModel), or the specific threshold to be 
        used for OOD detection.
        """
        pass
    
    def train(self, dataloader, *args, **kwargs) -> dict:
        """
        Trains the OOD detection algorithm using the specified `dataloader`.

        Args:
            dataloader (torch.utils.data.DataLoader): A dataloader that 
                returns (x, y) pairs (some OOD detection algorithms only 
                require x and no y, but yielding (x, y) pairs always is more 
                consistent across different OOD detection algorithms)
        
        Returns:
            A dictionary of training metrics.
        """
        pass
    
    def predict(self, dataloader, *args, **kwargs) -> np.ndarray:
        """
        Predicts which examples in the dataloader are OOD **in the order given**.
        This means that the dataloader should be initialized with shuffle=False so 
        that the OOD predictions are consistent across runs.

        NOTE: Please make sure that the dataloader has been initialized with
        shuffle=False, since this function returns predictions for each observation
        in the order that the observations were given.

        Args:
            dataloader (torch.utils.data.DataLoader): A dataloader that returns
                (x, y) pairs and which has been initialized with shuffle=False.
        
        Returns:
            A binary np.ndarray `arr` where arr[i] == 1 if the i-th example in 
            the dataloader is predicted to be OOD, otherwise arr[i] == 0.
        """
        pass

class Ensemble(OODDetector):
    def __init__(self, ood_detectors: list[OODDetector], method="intersection"):
        """
        Ensemble of OOD detectors.
        
        Parameters
        ----------
        ood_detectors : list[OODDetector]
            List of OOD detectors to ensemble.
        method : str, optional
            Method to use for ensembling, by default "intersection". Options:
            - "intersection": An observation is considered OOD if all
                detectors in the ensemble consider it OOD.
            - "union": An observation is considered OOD if any detector
                in the ensemble considers it OOD.
            - "majority": An observation is considered OOD if the majority of
                detectors in the ensemble consider it OOD.
        """
        if method not in ["intersection", "union", "majority"]:
            raise ValueError("Invalid method. Choose from 'intersection', 'union', or 'majority'.")
        
        self.ood_detectors = ood_detectors
        self.method = method
    
    def train(self, dataloaders, *args, **kwargs) -> dict:
        """
        Trains the OOD detection algorithm using the specified `dataloader`.

        Parameters
        ----------
        dataloaders : list
            A list of dataloaders that each returns (x, y) pairs.
        
        Returns
        -------
        dict
            A dictionary of training metrics.
        """
        for ood_detector, dataloader in zip(self.ood_detectors, dataloaders):
            ood_detector.train(dataloader, *args, **kwargs)
    
    def predict(self, dataloaders) -> np.ndarray:
        """
        A binary np.ndarray `arr` where arr[i] == 1 if the i-th example in
        the dataloaders is predicted to be OOD, otherwise arr[i] == 0. The
        dataloaders should index the same observations, but separate dataloaders
        may be required for each OOD detection method.
        """
        predictions = [ood_detector.predict(dataloader) for ood_detector, dataloader in zip(self.ood_detectors, dataloaders)]
        if self.method == "intersection":
            return np.all(predictions, axis=0).astype(int)
        elif self.method == "union":
            return np.any(predictions, axis=0).astype(int)
        elif self.method == "majority":
            return (np.sum(predictions, axis=0) > len(self.ood_detectors) / 2).astype(int)
