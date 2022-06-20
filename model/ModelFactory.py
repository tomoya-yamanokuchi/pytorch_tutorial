import imp
from .Mnist_CNN import Mnist_CNN
from .variational_autoencoder.Encoder import Encoder
from .variational_autoencoder.Decoder import Decoder
from .variational_autoencoder.VariationalAutoencoder import VariationalAutoencoder

class ModelFactory:
    def create(self, name: str, **kwargs):
        if      name == "mnist_cnn": return Mnist_CNN(**kwargs)
        elif    name == "encoder"  : return Encoder(**kwargs)
        elif    name == "decoder"  : return Decoder(**kwargs)
        elif    name == "vae"      : return VariationalAutoencoder(**kwargs)
        else                       : raise NotImplementedError()
