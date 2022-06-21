import torch.utils.data as data
from torchvision import datasets
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))


from model.LitAutoEncoder import *
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader


# Load data sets
train_transforms = transforms.Compose([transforms.ToTensor()])
train_set = datasets.MNIST(root="MNIST", transform=train_transforms, download=True, train=True)
test_set  = datasets.MNIST(root="MNIST", transform=train_transforms, download=True, train=False)

# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

train_set = DataLoader(train_set)
val_set = DataLoader(valid_set)

train_transforms = transforms.Compose([transforms.ToTensor()])

autoencoder = LitAutoEncoder(Encoder(), Decoder())

trainer = pl.Trainer()
trainer.fit(model=autoencoder, train_dataloaders=train_set, val_dataloaders=val_set)
trainer.test(model=autoencoder, dataloaders=DataLoader(test_set))

