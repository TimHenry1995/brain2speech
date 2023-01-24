import torch

class Sum(torch.nn.Module):
    def __init__(self) -> object:
        pass

    def __forward__(self, input):
        return torch.nn.sum(input=input)
