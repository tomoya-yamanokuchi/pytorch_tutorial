import torch
from torch import nn
from torch import Tensor
from typing import List
from .Encoder import Encoder
from .Decoder import Decoder


class VariationalAutoencoder(nn.Module):
    def __init__(self,
                 in_channels      : int,
                 conv_out_channels: int,
                 latent_dim       : List = None,
                 **kwargs) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, conv_out_channels, latent_dim)
        self.decoder = Decoder(in_channels, conv_out_channels, latent_dim)


    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param mu    : (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return      : (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encoder.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decoder.decode(z), input, mu, log_var]