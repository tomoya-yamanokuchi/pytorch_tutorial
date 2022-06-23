import torch
from torch import optim
from typing import List
import pytorch_lightning as pl
from .VariationalAutoencoder import VariationalAutoencoder


class LitVariationalAutoencoder(pl.LightningModule):
    def __init__(self,
                #  in_channels      : int,
                #  conv_out_channels: int,
                #  latent_dim       : List = None,
                 **kwargs) -> None:
        super().__init__()
        self.model = VariationalAutoencoder(**kwargs)


    def training_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device

        results    = self.model.forward(real_img)
        train_loss = self.model.loss_function(
            *results,
            M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
            batch_idx = batch_idx
        )
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        return train_loss['loss']


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
