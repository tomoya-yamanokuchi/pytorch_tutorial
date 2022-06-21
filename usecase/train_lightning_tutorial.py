import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from model.LitAutoEncoder import LitAutoEncoder, Encoder, Decoder

dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset)


# model
autoencoder = LitAutoEncoder(Encoder(), Decoder())

# train model
trainer = pl.Trainer(accelerator="gpu", devices=[0])
trainer.fit(model=autoencoder, train_dataloaders=train_loader)