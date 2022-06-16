import imp
from .Mnist_CNN import Mnist_CNN


class ModelFactory:
    def create(self, name: str):
        if name == "mnist_cnn": return Mnist_CNN()
        else                  : raise NotImplementedError()
