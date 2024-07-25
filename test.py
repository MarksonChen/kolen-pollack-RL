from kolen_pollack import KolenPollackMLP
import torch

model = KolenPollackMLP(2, [4], 1)
input = torch.ones(4, 3, 3)
print(input)
print(model(input))
