import time
from tracemalloc import start
import torch
from model.Mnist_CNN import Mnist_CNN
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from MNIST_data_setup import MNIST_data_setup

from dataloader.WrappedDataLoader import WrappedDataLoader
from dataloader.MNISTDataLoader import MNISTDataLoader

print(torch.cuda.is_available())

dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

# dev = torch.device("cpu")

mnist = MNIST_data_setup()
mnist.download()
((x_train, y_train), (x_valid, y_valid), _) = mnist.load()


x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)

bs = 256
# train_ds = TensorDataset(x_train, y_train)
# train_dl = DataLoader(train_ds, batch_size=bs)

# valid_ds = TensorDataset(x_valid, y_valid)
# valid_dl = DataLoader(valid_ds, batch_size=bs * 2)


# train_dl = WrappedDataLoader(train_dl, dev=dev)
# valid_dl = WrappedDataLoader(valid_dl, dev=dev)


train_dl = MNISTDataLoader((x_train, y_train), batch_size=bs, dev=dev)
valid_dl = MNISTDataLoader((x_valid, y_valid), batch_size=bs, dev=dev)


lr = 0.5
model = Mnist_CNN()

model.to(dev)

opt       = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
loss_func = F.cross_entropy
epochs    = 100



for epoch in range(epochs):
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