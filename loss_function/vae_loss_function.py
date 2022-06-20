import torch
from torch.nn import functional as F


def vae_loss_function(*args, **kwargs) -> dict:
    """
    Computes the VAE loss function KL(N(\mu, \sigma), N(0, 1)).
    """
    recons  = args[0]
    input   = args[1]
    mu      = args[2]
    log_var = args[3]

    pred, yb

    kld_weight  = kwargs['M_N'] # Account for the minibatch samples from the dataset
    recons_loss = F.mse_loss(recons, input)

    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    loss = recons_loss + kld_weight * kld_loss
    return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}