import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
import os
import torch
# from model.variational_autoencoder.LitVariationalAutoencoder import LitVariationalAutoencoder
from model.ModelFactory import ModelFactory
from torch import optim, nn, utils, Tensor
from torchvision import transforms
from MNIST_data_setup import MNIST_data_setup
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from dataloader.DataLoaderFactory import DataLoaderFactory

from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


class Train:
    def run(self, config):
        # autoencoder = LitVariationalAutoencoder(**config.model)

        model        = ModelFactory().create(**config.model)

        # dataset      = MNIST(os.getcwd(), download=True, transform= transforms.ToTensor())
        # train_loader = DataLoader(dataset)

        mnist = MNIST_data_setup()
        mnist.download()
        ((x_train, y_train), (x_valid, y_valid), _) = mnist.load()
        DataLoader = DataLoaderFactory().create(config.dataloader.name); print(DataLoader)
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # train_loader = DataLoader((x_train, y_train), config=config.dataloader, dev=dev)
        train_loader = DataLoader((x_valid, y_valid), config=config.dataloader, dev=dev)
        valid_loader = DataLoader((x_valid, y_valid), config=config.dataloader, dev=dev)

        tb_logger =  TensorBoardLogger(
            save_dir = config.logging_params.save_dir,
            name     = config.model.name,
        )

        trainer = pl.Trainer(
            gpus                = [0],
            limit_train_batches = 100,
            max_epochs          = 10,
            callbacks = [
                ModelCheckpoint(
                    # save_top_k = 2,]
                    dirpath                 = os.path.join(tb_logger.log_dir , "checkpoints"),
                    filename                = '{epoch}',
                    every_n_epochs          = 1,
                    monitor                 = "Reconstruction_Loss",
                    save_last               = False,
                    save_on_train_epoch_end = True,
                )
            ],
            logger=tb_logger,
        )

        # trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

        from dataloader.MNISTDataModule import MNISTDataModule
        data = MNISTDataModule("./")
        trainer.fit(model=model, datamodule=data)



if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../conf", config_name="config")
    def get_config(cfg: DictConfig) -> None:
        train = Train()
        train.run(cfg)

    get_config()