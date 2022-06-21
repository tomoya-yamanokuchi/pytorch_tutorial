import torch
from torch import Tensor
from torch import nn
from typing import List


class Decoder(nn.Module):
    def __init__(self,
                 in_channels      : int,
                 hidden_dims: List[int],
                 latent_dim       : int,
                 **kwargs) -> None:
        super().__init__()

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels    = hidden_dims[i],
                        out_channels   = hidden_dims[i + 1],
                        kernel_size    = 3,
                        stride         = 2,
                        padding        = 1,
                        output_padding = 1
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels    = hidden_dims[-1],
                    out_channels   = hidden_dims[-1],
                    kernel_size    = 3,
                    stride         = 2,
                    padding        = 1,
                    output_padding = 1
                ),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels  = hidden_dims[-1],
                    out_channels = 3,
                    kernel_size  = 3,
                    padding      = 1
                ),
                nn.Tanh()
            )
        )
        self.decoder = nn.Sequential(*modules)



    def forward(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        : return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        return result
