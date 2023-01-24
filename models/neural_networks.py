import torch
import torch.nn as nn
from abc import ABC
from sklearn.utils import shuffle
from typing import Union, List, Dict
import sys
sys.path.append(".")
from models import streamable_modules as streamable
from models import module_converter as mc

# Neural Network
class NeuralNetwork(torch.nn.Module, ABC):
    """This class provides functionality for neural networks that can be used in streamable and stationary mode."""

    def __init__(self, is_streamable: bool) -> object:
        """Constructor for this class. The inheriting subclasses are expected to behave as follows:
        0. Call this super initializer.
        1. Create a dictionary that gives for a module name its corresponding stationary module, e.g. convertible_modules = {'linear_1': torch.nn.Linear(...), 'linear_2': torch.nn.Linear(...)}
        2. Convert them to streamable modules if needed using: if is_streamable: convertible_modules = NeuralNetwork.__to_streamable__(modules=convertible_modules)
        3. Save the convertible modules such that they can be used for your later computations. 
            E.g. if you use a sequential module, provide them to the sequential module or 
            if you use the modules as attributes of your neural network, e.g. query, key, value 
            layers in an attention mechanism then use the ones from the convertible_modules dictionary.
        
        Inputs:
        - is_streamable: Indicates whether this neural network shall be used in streaming or stationary mode.
        """
        super(NeuralNetwork, self).__init__()

        self.__is_streamable__ = is_streamable

    @staticmethod
    def __to_streamable__(modules: Dict[str,torch.nn.Module]) -> Dict[str,streamable.Module]:
        """Converts modules from stationary to streaming. Assumes that the module_conveter has a mapping for that module.
        
        Inputs:
        - modules: provides for a module name key a stationary module, e.g. Linear, Conv1d, Sum. 
        
        Outputs:
        - modules: the same dictionary with its modules replaced by their streamable equivalents."""

        # Replace each module with its alternative
        for name, module in modules.items():
            modules[name] = mc.ModuleConverter.stationary_to_streamable(module=module)

        # Outputs
        return modules

    def predict(self, x: torch.Tensor, is_final_slice: bool = None) -> torch.Tensor:
        """With this function the trained model can be used to map from input to output.
        
        Inputs:
        - x: The input to the neural network. Shape == [time frame count, input feature count].
        - is_final_slice: optional in stationary mode, required in streamable mode. Indicates whether this is the final slice of the data stream. 

        Outputs:
        - y_hat: The output of the neural network. Shape == [time frame count, output feature count]."""
        
        # Input validity
        assert type(x) == torch.Tensor, f"Expected x to have type torch.Tensor, received {type(x)}."
        if self.__is_streamable__: assert is_final_slice != None, f"Input is_final_slice is required in streamable mode. Received {is_final_slice}."

        # Handle streamable mode
        if self.__is_streamable__ and is_final_slice: streamable.Module.final_slice()

        # Predict
        y_hat = self(x)

        # Handle streamable mode
        if self.__is_streamable__ and is_final_slice: streamable.Module.close_stream()

        # Outputs
        return y_hat

# Dense neural network
class Dense(NeuralNetwork):
    """This class is a dense neural network."""

    def __init__(self, input_feature_count: int, output_feature_count: int, is_streamable: bool = False) -> object:
        """Constructor for this class.
        
        Inputs:
        - input_feature_count: number of input features.
        - output_feature_count: number of output features.
        - is_streamable: Indicates whether this neural network shall be used in streaming or stationary mode.
        """
    
        # Following the super class instructions for initialization
        # 0. Super
        super(Dense, self).__init__(is_streamable=is_streamable)

        # 1. Create a dictionary of stationary modules
        convertible_modules = { 'linear_1': nn.Linear(input_feature_count, output_feature_count),
                                'linear_2': nn.Linear(output_feature_count, output_feature_count)}
        
        # 2. Convert them to streamable modules
        if is_streamable: convertible_modules = NeuralNetwork.__to_streamable__(modules=convertible_modules)
        
        # 3. Save the convertible modules for later computations     
        self.sequential = nn.Sequential(convertible_modules['linear_1'], nn.ReLU(), convertible_modules['linear_2'])
  
    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Predict
        y_hat = self.sequential(x)

        # Outputs
        return y_hat
        