from torch.nn import functional
from .vae_loss_function import vae_loss_function

class LossFunctionFactory:
    def create(self, name: str):
        if name == "vae_loss_function": return vae_loss_function
        else                          : return functional.__dict__[name]

if __name__ == '__main__':
    loss_func = LossFunctionFactory().create("mse_loss")
    print(loss_func)

