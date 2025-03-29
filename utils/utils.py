from typing import Union

import numpy as np
import pandas as pd

import torch

class DualWriter:
    def __init__(self, *writers):
        """
        Writer class for sys.stdout setting. This class will write to any number 
        of possible writers, include stdout and a file. This also works for
        stderr.

        Examples:
            Writing to both stdout and a file automatically (no changes to print statements):
            ```
            STDOUT_WRITE_FILE = open(os.path.join(args.save, "stdout.txt"), "w")
            dual_writer_stdout = utils.DualWriter(sys.stdout, STDOUT_WRITE_FILE)
            sys.stdout = dual_writer_stdout
            ...
            STDOUT_WRITE_FILE.close()
            ```
        """
        self.writers = writers

    def write(self, message):
        for writer in self.writers:
            writer.write(message)
        
    def flush(self):
        for writer in self.writers:
            writer.flush()

class CategoricalMapper:
    def __init__(self, data: Union[np.ndarray, pd.Series], allow_unknown: bool=False):
        """
        Maps categorical data to non-negative integers, with the option to handle
        querying of unknown classes. This class can be indexed like a dictionary or
        called like a function.

        Arguments
        ----------
        data : np.ndarray or pd.Series
            The categorical data to be mapped. Values don't need to be unique.
        allow_unknown : bool, optional
            Whether or not to allow unknown classes to be indexed at test time.
        """
        if any([np.isnan(x) for x in data if isinstance(x, float)]):
            raise ValueError("NaN values are not allowed in the data, as this will break __getitem__.")

        self.allow_unknown = allow_unknown

        self.known_keys = np.unique(data)
        self.known_values = np.arange(len(self.known_keys))

        if self.allow_unknown:
            self.unknown_value = len(self.known_keys)

    def __len__(self):
        """
        Number of unique categories.
        """
        return len(self.known_keys) + (1 if self.allow_unknown else 0)
    
    def keys(self):
        return np.copy(self.known_keys)
    
    def values(self):
        return np.copy(self.known_values)

    def __getitem__(self, key) -> int:
        """
        Returns the numerical value corresponding to the given key. If the key is 
        not found and allow_unknown is True, returns the unknown_value.
        """
        try:
            return int(self.known_values[np.where(self.known_keys == key)[0][0]])
        except IndexError:
            if self.allow_unknown:
                return self.unknown_value
            else:
                raise KeyError(f"{key} not found in known keys.")

    def __call__(self, x):
        return self[x]