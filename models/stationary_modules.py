import torch

class Sum(torch.nn.Module):
    def __init__(self) -> object:
        super(Sum, self).__init__()

    def forward(self, input):
        return input[0] + input[1]

class GRU(torch.nn.Module):
    """This class provides a gated recurrent unit that does not output the state."""

    def __init__(self, **kwargs) -> object:
      """Constructor for this class.
      
      Inputs:
      - kwargs: The same as torch.nn.GRU.
      
      Outputs:
      - The instance of this class."""
      super(GRU, self).__init__()

      # GRU layers
      self.gru = torch.nn.GRU(**kwargs)
      
    def forward(self, x: torch.Tensor) -> torch.Tensor:
      """Executes the forward operation of this module. Does not return the state.
      
      Inputs:
      - x: Input tensor of shape [instances per batch, time frames per instance, input feature count].
      
      Outputs:
      - y_hat: Prediction of shape [instances per batch, time frames per instance, output feature count]"""

      # Forward
      y_hat, _ = self.gru(x)
      
      # Outputs
      return y_hat