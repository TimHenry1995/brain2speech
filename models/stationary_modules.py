import torch

class Sum(torch.nn.Module):
    def __init__(self) -> object:
        super(Sum, self).__init__()

    def forward(self, input):
        return input[0] + input[1]
