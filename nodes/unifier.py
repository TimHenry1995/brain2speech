import numpy as np
from timeflux.core.node import Node
import collections

class Unifier(Node):
    """This node collects inputs from the ports for EEG, speech and labels. It buffers them and always outputs a data frame that contains
    the common time frames of these inputs."""
    

    def __init__(self) -> object:
        """Constructor for this class.
        
        Inputs:
        - None
        
        Outputs:
        - self: initialized instance of this class."""

        self.__x_buffer__ = collections.deque()
        self.__y_buffer__ = collections.deque()
        self.__labels_buffer__ = collections.deque()
        self.__time_frames_in_x_buffer__ = 0
        self.__time_frames_in_y_buffer__ = 0
        self.__time_frames_in_labels_buffer__ = 0

    def update(self):
        for _, _, port in self.iterate("i*"):
            if (port.ready()):
                # Notify user
                self.logger.info(f"Acquired data from stream: {port.meta.get('stream_name')} with shape: {port.data.shape}.")
                """
                if (self._buffer is not None):
                    self.o = port

                    self.logger.info('Acquired second data of type: {} (shape: {})'
                                     .format(self.o.meta.get('d_type'), self.o.data.shape))

                    # this logic ensures that the first value of the output array
                    # is always in the order output[0]=eeg, output[1]=audio.
                    output = []
                    if (self._buffer_meta.get('d_type') == 'audio'):
                        output.append(port.data)
                        output.append(self._buffer)
                    else:
                        output.append(self._buffer)
                        output.append(port.data)

                    # mix both metadata. This is currently used because only the eeg metadata
                    # contains the information about the 'eeg_bad_channels_indices'.
                    self._buffer_meta = {**self._buffer_meta, **port.meta}

                    # define the output meta
                    self._buffer_meta['d_type'] = 'eeg_audio'
                    # this is only valid for the eeg data because it is used for accumulating the
                    # windowed eeg features. So at this point the decision is to either delete this
                    # metadata or to change its value to something unharmful like 'None'.
                    self._buffer_meta['gate_status'] = None

                    # NOTE: I needed to add the line:
                    np.warnings.filterwarnings(
                        'ignore', category=np.VisibleDeprecationWarning)
                    # because the following warning was raised (and I didn't wanted to worry about this now):
                    # maybe later...
                    # numpy.VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences
                    # (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes)
                    # is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.

                    # set the output with the proper data and metadata
                    # define the output always with the eeg data first.
                    self.o.set([output[0], output[1]], meta=self._buffer_meta)
                    self.logger.info(
                        'Unification completed, single Dataframe placed on the output')

                else:
                    self._buffer = port.data
                    self._buffer_meta = port.meta
                    self.logger.info('Acquired first data of type: {} (shape: {})'
                                     .format(self._buffer_meta.get('d_type'), self._buffer.shape))

            """