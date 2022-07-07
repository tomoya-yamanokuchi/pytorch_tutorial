from turtle import color
import matplotlib
import torch
from torchvision import utils
from torch import Tensor
from torch import optim
from typing import List
import pytorch_lightning as pl
from .VariationalAutoencoder import VariationalAutoencoder
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib import ticker, cm



class LitVariationalAutoencoder(pl.LightningModule):
    def __init__(self,
                #  in_channels      : int,
                #  conv_out_channels: int,
                #  latent_dim       : List = None,
                kld_weight,
                 **kwargs) -> None:
        super().__init__()
        self.kld_weight = kld_weight
        self.model = VariationalAutoencoder(**kwargs)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


    def sample(self,
               num_samples:int,
               current_device: int,
               **kwargs) -> Tensor:
        z       = torch.randn(num_samples, self.latent_dim)
        z       = z.to(current_device)
        samples = self.model.decoder.forward(z)
        return samples


    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.model.forward(x)[0]


    def training_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device

        results    = self.model.forward(real_img)
        train_loss = self.model.loss_function(
            *results,
            M_N = self.kld_weight, #al_img.shape[0]/ self.num_train_imgs,
            batch_idx = batch_idx
        )
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        return train_loss['loss']


    def validation_step(self, batch, batch_idx):
        # import numpy as np
        # a = np.random.rand(3)
        # np.save("aaaaaa.npy", a)
        real_img, labels = batch
        results     = self.model.forward(real_img)
        recons      = results[0][0]
        input       = results[1][0]

        mu          = results[2] # Size([num_batch, dim_z])
        log_var     = results[3] # Size([num_batch, dim_z])

        utils.save_image(
            torch.cat([recons, input], dim=2),
            'sample_' + str(self.current_epoch) + '.png',
        )
        self.visualize_samples(mu, labels)


    def visualize_samples(self, data, label):
        if isinstance(data, Tensor):
            data = data.cpu().numpy()
        if isinstance(label, Tensor):
            label = label.cpu().numpy()

        plt.figure(figsize=(4, 4))
        plt.scatter(data[:, 0], data[:, 1], edgecolor="#333", c=label, cmap="jet")
        # plt.scatter(data[:, 0], data[:, 1], edgecolor="#333", label="Class 1")
        plt.title("Dataset samples")
        plt.ylabel(r"$x_2$")
        plt.xlabel(r"$x_1$")
        # plt.legend()
        # plt.show()
        plt.savefig('latentn_space_' + str(self.current_epoch) + '.png')