import numpy
from torch import device
from typing import Tuple
from omegaconf import DictConfig
from .BaseWrappedDataLoader import BaseWrappedDataLoader


class MNISTDataLoader(BaseWrappedDataLoader):
    def __init__(self, data: Tuple[numpy.ndarray], config: DictConfig, dev: device):
        super().__init__(data, config, dev)


    def preprocess(self, x, y):
        if self.dev is None: return x.view(-1, 1, 28, 28), y
        else               : return x.view(-1, 1, 28, 28).to(self.dev), y.to(self.dev)