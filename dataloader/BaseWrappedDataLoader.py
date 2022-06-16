import numpy
import torch
from typing import Tuple
from torch import device
from torch import Tensor
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from omegaconf import DictConfig


class BaseWrappedDataLoader:
    def __init__(self,  data: Tuple[numpy.ndarray], config: DictConfig, dev: device):
        ds        = TensorDataset(*tuple(map(torch.tensor, data)))
        self.dl   = DataLoader(
            dataset     = ds,
            batch_size  = config.batch_size,
            shuffle     = config.shuffle,
            num_workers = config.num_workers,
            pin_memory  = config.pin_memory
        )
        self.dev  = dev
        self.func = self.preprocess


    def __len__(self):
        return len(self.dl)


    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


    def preprocess(self, *tensor: Tensor):
        return tensor
