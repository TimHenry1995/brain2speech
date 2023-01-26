import numpy as np
from timeflux.core.node import Node
from typing import List
import pandas as pd

class Unifier(Node):
    """This node collects inputs from several ports. It buffers them and outputs a a tuple that contains
    dataframes with the common time frames of these inputs."""
    

    def __init__(self, stream_names: List[str]) -> object:
        """Constructor for this class.
        
        Inputs:
        - stream_names: Assumed to be non-empty list of unique strings. The names of the streams. Each such name is used to identify a port and save its data in its corresponding buffer.
        
        Outputs:
        - self: initialized instance of this class."""

        # Initialize buffers
        self.__buffers__ = {stream_name: pd.DataFrame() for stream_name in stream_names}
        
    def update(self):
        for _, _, port in self.iterate("i*"):
            if (port.ready()):
                # Input validity
                assert port.meta.get('stream_name') in self.__buffers__.keys(), f"Expected stream_name meta variable of input to be in {self.__buffers__.keys()}, but received{port.meta.get('stream_name')}"

                # Append to buffer
                self.__buffers__[port.meta.get('stream_name')] = pd.concat(objs=[self.__buffers__[port.meta.get('stream_name')], port.data])
                
        # Get the number of common time frames
        lengths = [None] * len(self.__buffers__.keys())
        for index, buffer in enumerate(self.__buffers__.values()):
            lengths[index] = len(buffer)
        common_time_frame_count = np.min(lengths)
        
        # Output the leading common time frames
        output = {name: buffer.iloc[:common_time_frame_count] for name, buffer in self.__buffers__.items()}
        self.o.data = output

        # Save only the remaining time points
        for name, buffer in self.__buffers__.items():
            self.__buffers__[name] = buffer.iloc[common_time_frame_count:]