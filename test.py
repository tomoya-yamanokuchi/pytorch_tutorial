import torch


x = torch.linspace(1, 3, 3).reshape(3, 1).tile(1, 5)

print(x)
print()
print(x.transpose(1, 0))


