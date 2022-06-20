from pathlib import Path
import requests
import pickle
import gzip
from matplotlib import pyplot as plt


class MNIST_data_setup:
    def __init__(self):
        DATA_PATH     = Path("data")
        self.PATH     = DATA_PATH / "mnist"
        self.PATH.mkdir(parents=True, exist_ok=True)
        self.URL      = "https://github.com/pytorch/tutorials/raw/master/_static/"
        self.FILENAME = "mnist.pkl.gz"


    def download(self):
        if not (self.PATH / self.FILENAME).exists():
            content = requests.get(self.URL + self.FILENAME).content
            (self.PATH / self.FILENAME).open("wb").write(content)


    def load(self):
        with gzip.open((self.PATH / self.FILENAME).as_posix(), "rb") as f:
            ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
        return ((x_train, y_train), (x_valid, y_valid), _)


    def show_data(self, x):
        plt.imshow(x[0].reshape((28, 28)), cmap="gray")
        plt.show()
        print(x.shape)


if __name__ == '__main__':

    mnist = MNIST_data_setup()
    mnist.download()
    ((x_train, y_train), (x_valid, y_valid), _) = mnist.load()

    mnist.show_data(x_train)


    import torch
    print(x_train, y_train)
    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )
    n, c = x_train.shape
    print(x_train, y_train)
    print(x_train.shape)
    print(y_train.min(), y_train.max())