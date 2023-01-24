import sys
sys.path.append(".")
from models import streamable_modules as streamable
from models import stationary_modules as stationary
import torch
from abc import ABC
from typing import Type

class ModuleConverter(ABC):
    """This abstract base class defines methods for converting between stationary and streamable modules.
    The interiting subclasses must override the two methods stationary_to_streamable and streamable_to_stationary 
    as they will be called by this base class."""

    @staticmethod
    def stationary_to_streamable(module: torch.nn.Module) -> streamable.Module:
        """Converts a stationary module to a streamable module. The module attributes may be copied by reference. 
        The streamable module will start with its default state.
        
        Inputs:
        - module: The stationary module.
        
        Outputs:
        - streamable_module: The streamable equivalent."""

        # Extract the name of the module type
        module_name = str(type(module)).split(".")[-1][:-2] # Taking the lowest level class name (-1) and removing the trailing two characters which are '>
        
        # Construct the type of the converter
        converter_name = f"{module_name}Converter"
        converter_type = globals().get(converter_name) # The running python process has a globals dictionary that holds the converter class
        assert f"module_converter.{converter_name}" in str(converter_type), f"Unable to convert module from stationary to streamable because the required converter called {type(module)}Converter could not be found. It should suffice to declare such a corresponding converter class here and make sure no other package loads a class with the same name into the globals."
        
        # Convert
        streamable_module = converter_type.stationary_to_streamable(module=module)

        # Outputs
        return streamable_module

    @staticmethod
    def streamable_to_stationary(module: streamable.Module) -> torch.nn.Module:
        """Converts a streamable module to a stationary module. The module attributes may be copied by reference. 
        
        Inputs:
        - module: The streamable module.
        
        Outputs:
        - stationary_module: The stationary equivalent."""
            
        # Extract the name of the module type
        module_name = str(type(module)).split(".")[-1][:-2] # Taking the lowest level class name (-1) and removing the trailing two characters which are '>
        
        # Construct the type of the converter
        converter_name = f"{module_name}Converter"
        converter_type = globals().get(converter_name) # The running python process has a globals dictionary that holds the converter class
        assert f"module_converter.{converter_name}" in str(converter_type), f"Unable to convert module from streamable to stationary because the required converter called {type(module)}Converter could not be found. It should suffice to declare such a corresponding converter class here and make sure no other package loads a class with the same name into the globals."
        
        # Convert
        stationary_module = converter_type.streamable_to_stationary(module=module)

        # Outputs
        return stationary_module\

    @staticmethod
    def __convert__(module: object, target_type: Type) -> object:
        """Maps the attributes from the given module to a new instance of target type.
        
        Inputs:
        - module: The module for which the attributes shall be copied.
        - target_type: The type of the new module.
        
        Outputs:
        - new_module: The new module of target type with the attributes of the old one."""
        raise NotImplementedError()

class Conv1dConverter(ModuleConverter):
    """This class provides a method to convert between stationary and streamable Conv1d."""

    @staticmethod
    def __convert__(module: object, target_type: Type) -> object:
        # Create module equivalent
        new_module = target_type(in_channels = module.in_channels,
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
        new_module.weight = module.weight
        new_module.bias = module.bias

        # Outputs
        return new_module

    @staticmethod
    def stationary_to_streamable(module: torch.nn.Module) -> streamable.Module:
        # Create module equivalent
        streamable_module = Conv1dConverter.__convert__(module= module, target_type=streamable.Conv1d)

        # Outputs
        return streamable_module

    @staticmethod
    def streamable_to_stationary(module: streamable.Module) -> torch.nn.Module:
        # Create module equivalent
        stationary_module = Conv1dConverter.__convert__(module= module, target_type=torch.nn.Conv1d)

        # Outputs
        return stationary_module

class ConvTranspose1dConverter(ModuleConverter):
    """This class provides a method to convert between stationary and streamable ConvTranspose1d."""

    @staticmethod
    def __convert__(module: object, target_type: Type) -> object:
        # Input validity
        assert module.output_padding == 0, f"Any non-zero value for output_padding is not supported. Received {module.output_padding}."

        # Create module equivalent
        new_module = target_type(in_channels = module.in_channels,
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
        new_module.weight = module.weight
        new_module.bias = new_module.bias

        # Outputs
        return new_module

    @staticmethod
    def stationary_to_streamable(module: torch.nn.Module) -> streamable.Module:
        # Create module equivalent
        streamable_module = ConvTranspose1dConverter.__convert__(module=module, target_type=streamable.ConvTranspose1d)

        # Outputs
        return streamable_module

    
    @staticmethod
    def streamable_to_stationary(module: streamable.Module) -> torch.nn.Module:
        # Create module equivalent
        stationary_module = ConvTranspose1dConverter.__convert__(module=module, target_type=torch.nn.ConvTranspose1d)

        # Outputs
        return stationary_module
        

class SumConverter(ModuleConverter):
    """This class provides a method to convert between stationary and streamable Sum."""

    @staticmethod
    def stationary_to_streamable(module: torch.nn.Module) -> streamable.Module:
        # Create module equivalent
        streamable_module = streamable.Sum()

        # Outputs
        return streamable_module

    @staticmethod
    def streamable_to_stationary(module: streamable.Module) -> torch.nn.Module:
        # Create module equivalent
        stationary_module = stationary.Sum()

        # Outputs
        return stationary_module

class ReflectionPad1dConverter(ModuleConverter):
    """This class provides a method to convert between stationary and streamable ReflectionPad1d."""

    @staticmethod
    def stationary_to_streamable(module: torch.nn.Module) -> streamable.Module:
        # Create module equivalent
        streamable_module = streamable.ReflectionPad1d(padding = module.padding)

        # Outputs
        return streamable_module

    @staticmethod
    def streamable_to_stationary(module: streamable.Module) -> torch.nn.Module:
        # Create module equivalent
        stationary_module = torch.nn.ReflectionPad1d(padding = module.padding)

        # Outputs
        return stationary_module

class LinearConverter(ModuleConverter):
    """This class provides a method to convert between stationary and streamable Linear."""

    @staticmethod
    def __convert__(module: object, target_type: Type) -> object:
        # Create module equivalent
        new_module = target_type(
            kwargs={"in_features": module.in_features,
            "out_features": module.out_features,
            "bias": module.bias != None}
        )

        # Transfer the parameters
        new_module.weight = module.weight
        new_module.bias = module.bias

        # Outputs
        return new_module

    @staticmethod
    def stationary_to_streamable(module: torch.nn.Module) -> streamable.Module:
        # Create module equivalent
        streamable_module = LinearConverter.__convert__(module=module, target_type=streamable.Linear)

        # Outputs
        return streamable_module

    @staticmethod
    def streamable_to_stationary(module: streamable.Module) -> torch.nn.Module:
        # Create module equivalent
        stationary_module = LinearConverter.__convert__(module=module, target_type=torch.nn.Linear)

        # Outputs
        return stationary_module