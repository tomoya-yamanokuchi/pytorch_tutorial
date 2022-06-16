from torch.nn import functional

class LossFunctionFactory:
    def create(self, name: str):
        return functional.__dict__[name]

if __name__ == '__main__':
    loss_func = LossFunctionFactory().create("mse_loss")
    print(loss_func)

