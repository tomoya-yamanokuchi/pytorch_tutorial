import time
import torch
from Mnist_CNN import Mnist_CNN
from torch import optim
from torch.utils.data import TensorDataset

from MNIST_data_setup import MNIST_data_setup

from dataloader.DataLoaderFactory import DataLoaderFactory
from dataloader.WrappedDataLoader import WrappedDataLoader
from dataloader.MNISTDataLoader import MNISTDataLoader
from optimizer.OptimizerFactory import OptimizerFactory
from loss_function.LossFunctionFactory import LossFunctionFactory

print(torch.cuda.is_available())


class Train:
    def run(self, config):
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # dev = torch.device("cpu")

        mnist = MNIST_data_setup()
        mnist.download()
        ((x_train, y_train), (x_valid, y_valid), _) = mnist.load()

        DataLoader = DataLoaderFactory().create(config.dataloader.name); print(DataLoader)
        train_dl   = DataLoader((x_train, y_train), config=config.dataloader, dev=dev)
        valid_dl   = DataLoader((x_valid, y_valid), config=config.dataloader, dev=dev)

        model = Mnist_CNN()
        model.to(dev)

        opt = OptimizerFactory().create(model.parameters(), **config.optimizer)
        print(opt)

        loss_func = LossFunctionFactory().create(config.loss_function)

        for epoch in range(config.epoch):
            start_time = time.time()
            model.train()
            for xb, yb in train_dl:
                pred = model(xb)
                loss = loss_func(pred, yb)

                loss.backward()
                opt.step()
                opt.zero_grad()

            model.eval()
            with torch.no_grad():
                valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

            print(epoch, valid_loss, time.time() - start_time)


if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="conf", config_name="config")
    def get_config(cfg: DictConfig) -> None:
        train = Train()
        train.run(cfg)

    get_config()