import sys
sys.path.append(".")
from models import streamable_modules as streamable
import torch
from abc import ABC

class ModuleConverter():
    """This class defines methods for converting between stationary and streamable modules."""

    @staticmethod
    def stationary_to_streamable(module: torch.nn.Module) -> streamable.Module:
        """Converts a stationary module to a streamable module. The module attributes may be copied by reference. 
        
        Inputs:
        - module: The stationary module.
        
        Outputs:
        - module: The streamable equivalent."""

        # Extract stationary type
        stationary_type = str(type(module)).split(".")[-1][:-2] # Taking the lowest level class name (-1) and removing the trailing two characters which are '>
        to_streamable_type = globals().get("ToStreamable"+stationary_type) # Takes the type from the globals
        assert f"module_converter.ToStreamable{stationary_type}" in str(to_streamable_type), f"Unable to find the streamable equivalent for the given stationary module. Ensure the globals dictionary contains the streamable class corresponding to {type(module)}. It should suffice to declare such a corresponding class here and make sure no other package loads a class with the same name into the globals."

        # Convert
        module = to_streamable_type.stationary_to_streamable(module)

        # Outputs
        return module

class ToStreamable(ABC):

    @staticmethod
    def stationary_to_streamable(module: torch.nn.Module) -> streamable.Module:
        """Converts a stationary module to a streamable module. The module attributes may be copied by reference. 
        
        Inputs:
        - module: The stationary module.
        
        Outputs:
        - module: The streamable equivalent."""

        raise NotImplementedError

class ToStreamableConv1d(ToStreamable):
    """This class provides a method to convert between stationary and streamable Conv1d."""

    @staticmethod
    def stationary_to_streamable(module: torch.nn.Module) -> streamable.Module:
        # Create streamable module equivalent
        streamable_module = streamable.Conv1d(in_channels = module.in_channels,
                out_channels = module.out_channels,
                kernel_size = module.kernel_size,
                stride = module.stride,
                padding = module.padding,
                dilation = module.dilation,
                groups = module.groups,
                bias = module.bias,
                padding_mode = module.padding_mode
            )

        # Transfer the parameters
        streamable_module.parameters = module.parameters

        # Outputs
        return streamable_module

class ToStreamableConvTranspose1d(ToStreamable):
    """This class provides a method to convert between stationary and streamable ConvTranspose1d."""

    @staticmethod
    def stationary_to_streamable(module: torch.nn.Module) -> streamable.Module:
        # Input validity
        assert module.output_padding == 0, f"Any non-zero value for stationary_module.output_padding is not supported. Received {module.output_padding}."

        # Create streamable module equivalent
        streamable_module = streamable.ConvTranspose1d(in_channels = module.in_channels,
                out_channels = module.out_channels,
                kernel_size = module.kernel_size,
                stride = module.stride,
                padding = module.padding,
                dilation = module.dilation,
                groups = module.groups,
                bias = module.bias,
                padding_mode = module.padding_mode
            )

        # Transfer the parameters
        streamable_module.parameters = module.parameters

        # Outputs
        return streamable_module

class ToStreamableSum(ToStreamable):
    """This class provides a method to convert between stationary and streamable Sum."""

    @staticmethod
    def stationary_to_streamable(module: torch.nn.Module) -> streamable.Module:
        # Create streamable module equivalent
        streamable_module = streamable.Sum()

        # Outputs
        return streamable_module

class ToStreamableReflectionPad1d(ToStreamable):
    """This class provides a method to convert between stationary and streamable ReflectionPad1d."""

    @staticmethod
    def stationary_to_streamable(module: torch.nn.Module) -> streamable.Module:
        # Create streamable module equivalent
        streamable_module = streamable.ReflectionPad1d(padding = module.padding)

        # Outputs
        return streamable_module

class ToStreamableLinear(ToStreamable):
    """This class provides a method to convert between stationary and streamable Linear."""

    @staticmethod
    def stationary_to_streamable(module: torch.nn.Module) -> streamable.Module:
        # Create streamable module equivalent
        streamable_module = streamable.Linear(
            kwargs={"in_features": module.in_features,
            "out_features": module.out_features,
            "bias": module.bias != None}
        )

        # Transfer the parameters
        streamable_module.weight = module.weight
        streamable_module.bias = module.bias

        # Outputs
        return streamable_module
