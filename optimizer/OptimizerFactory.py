from omegaconf import DictConfig
from torch import optim

class OptimizerFactory:
    def create(self, params, name: str, **kwargs):
        return optim.__dict__[name](params, **kwargs)
