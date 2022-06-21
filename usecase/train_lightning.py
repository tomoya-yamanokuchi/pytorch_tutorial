import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
import os
# from model.variational_autoencoder.LitVariationalAutoencoder import LitVariationalAutoencoder
from model.ModelFactory import ModelFactory
from torch import optim, nn, utils, Tensor

from torchvision.datasets import MNIST

import pytorch_lightning as pl

class Train:
    def run(self, config):
        # autoencoder = LitVariationalAutoencoder(**config.model)

        model      = ModelFactory().create(**config.model)

        dataset = MNIST(os.getcwd(), download=True)
        train_loader = utils.data.DataLoader(dataset)

        trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
        trainer.fit(model=model, train_dataloaders=train_loader)

if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../conf", config_name="config")
    def get_config(cfg: DictConfig) -> None:
        train = Train()
        train.run(cfg)

    get_config()