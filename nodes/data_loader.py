import pandas as pd
import numpy as np
from timeflux.core.node import Node
import os
import time

class DataLoader(Node):
    """This node loads data. It does not expect an input stream. It has one output stream which is a 
    dataframe of temporal slice of the matrix stored at the specified path. The time axis is assumed to be the 0th axis."""

    def __init__(self, name: str, data_folder_path: str, subject_identifier: str, stream_name: str, seconds_per_time_frame: float, proportion: float) -> object:
        """Constructor for this class.
        
        Inputs:
        - name: The name assigned to this node.
        - data_folder_path: The path to the folder where the data is stored.
        - subject_identifier: The identifier of the subejct, e.g. sub-01.
        - stream_name: The name of the stream, e.g. feat, procWords or spec.
        - seconds_per_time_frame: The number of seconds per time frame. Determines the number of time frames per slice.
        - proportion: Float in range [0,1] The proportion of total timeframes to be read.
        
        Outputs:
        - self: The initialized instance."""
        
        # Super
        super(DataLoader, self).__init__()

        # Copy fields
        self.name = name
        self.seconds_per_time_frame = seconds_per_time_frame
        self.__file_path__ = os.path.join(data_folder_path, f"{subject_identifier}_{stream_name}.npy")
        self.__stream_name__ = stream_name

        # Load data
        data = pd.DataFrame(np.load(self.__file_path__))

        # Save the proportion of interest
        f = (int)(proportion*len(data))
        self.__data__  = data.iloc[:f]

        # Set remaining fields
        self.__index__ = 0
        self.tick = time.time()

    def update(self) -> None:
        # Time since last call
        tock = time.time()
        seconds_since_last_update = tock-self.tick
        self.tick = tock

        # Size for current slice
        time_frames_of_slice = (int)(round(number=seconds_since_last_update/self.seconds_per_time_frame, ndigits=0))

        # Set the output
        self.o.data = self.__data__.iloc[self.__index__:self.__index__ + time_frames_of_slice]
        self.o.meta = {"file_path": self.__file_path__, "stream_name": self.__stream_name__}

        # Log
        if self.__index__ < len(self.__data__) and len(self.__data__) <= self.__index__  + time_frames_of_slice: 
            self.logger.info(f"{self.name} finished stream")  

        # Update internal index
        self.__index__ += time_frames_of_slice

    def terminate(self):
        return super().terminate()