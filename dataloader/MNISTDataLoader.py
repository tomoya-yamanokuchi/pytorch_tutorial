import numpy
from torch import device
from torchvision import transforms
from typing import Tuple
from omegaconf import DictConfig
from .BaseWrappedDataLoader import BaseWrappedDataLoader



class MNISTDataLoader(BaseWrappedDataLoader):
    def __init__(self, data: Tuple[numpy.ndarray], config: DictConfig, dev: device):
        super().__init__(data, config, dev)


    def preprocess(self, x, y):
        x = x.view(-1, 1, 28, 28)
        # x = transforms.Resize(size=64)(x)
        # x = self.convert_to_rgb(x)
        if self.dev is None: return x, y
        else               : return x.to(self.dev), y.to(self.dev)


    def convert_to_rgb(self, x):
        return x.repeat(1, 3, 1, 1)