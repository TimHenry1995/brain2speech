from timeflux.core.node import Node
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from typing import List, Any
import torch, os, sys, time
from typing import List
sys.path.append(".")
from models import neural_networks, utilities, fitter as mf
# importing module
import collections
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection

class Fitter(Node):
    """This node manages the flow of incoming data to a models.fitter.Fitter object which runs on a parallel process.
    It expects a single input stream of a tuple that contains time courses for eeg features, speech spectrogram and label. 
    It has one output stream which is a data frame with columns for mean train and validation loss."""

    def __init__(self, name: str, min_time_frames_in_buffer: int, eeg_stream_name: str, speech_stream_name: str, label_stream_name: str, neural_network_type: str, parameters_path: str, start_from_prefit: bool, skip: bool) -> object:
        """Constructor for this class.
        
        Inputs:
        - name: The name assigned to this node.
        - min_time_frames_in_buffer: The minimum number of time frames 
            that have to be in the buffer in order to send them to the models.Fitter object for fitting.
        - eeg_stream_name, speech_stream_name, label_stream_name: The names of the streams. Each such name is used to identify a stream from the input ports.
        - neural_network_type: The type of the neural network to be trained, e.g. Dense or Convolutional. This type will be taken from models.neural_networks.
        - parameters_path: The path to the file where the neural network parameters shall be stored after each fit operation. This path is relative to the parameters directory.
        - start_from_prefit: Indicates whether the neural network should load the network parameters from parameters_path before the first fit iteration. If set to False then fitting starts from random parameters.
        - skip: Indicates whether the fitting shall be skipped. If True then no output is given in update().
        """

        # Super
        super(Fitter, self).__init__()

        # Copy attributes
        self.name = name
        self.min_time_frames_in_buffer = min_time_frames_in_buffer
        self.eeg_stream_name = eeg_stream_name
        self.speech_stream_name = speech_stream_name
        self.label_stream_name = label_stream_name
        self.__skip__ = skip
        parameters_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'parameters', parameters_path + '.pt'))
        
        # Set the buffer
        self.__reset_buffer__()

        if not skip:
            # Set up parallel process
            self.__pipe_end_point__, other_end_point = Pipe()
            self.__parallel_process__ = Process(target=self.parallel_fitter, args=([other_end_point, neural_network_type, parameters_path, start_from_prefit]))
            self.__parallel_process__.daemon = True # Ensures that the parallel process is killed when the main process is killed.
            self.__parallel_process__.start()
            self.__parallel_process_is_busy__ = False # The other process is only considered busy when it is currently processing data

            # Initialize output
            self.__mean_train_loss__ = 0
            self.__mean_validation_loss__ = 0

    def update(self) -> None:
        """Maintains a buffer of EEG, spectrogram and labels and submits it regularly to a models.fitter.Fitter object. 
        Expects to receive input in the form of a dictionary with keys for the eeg, speech and label streams and values for 
        their respective data frames. As data streams into this node it will be stored in a buffer. 
        Once the number of received time frames surpasses self.min_time_frames_in_buffer the buffer is sent to a models.fitter.Fitter object which
        will fit a model on them on a parallel process. While the models.fitter.Fitter is busy this update function will buffer the new slices of
        incoming data. As soon as the buffer surpasses self.min_time_frames_in_buffer the fitter node will take
        the next opportunity to send it to the models.fitter.Fitter and the cycle continues."""
        
        # Set meta
        self.o.meta = {'stream_name': 'losses'}

        # Exit early
        if self.__skip__ or not self.i.ready(): 
            self.o.data = pd.DataFrame({'train': [0], 'validation': [0]})
            return
            
        # Take out x, y, labels
        eeg_slice = torch.Tensor(self.i.data[self.eeg_stream_name].values)
        speech_slice = torch.Tensor(self.i.data[self.speech_stream_name].values)
        label_slice = self.i.data[self.label_stream_name].iloc[:,0].values.tolist()
        
        # Extend the buffer by the input
        if len(label_slice) > 0: 
            self.__extend_buffer__(eeg_slice=eeg_slice, speech_slice=speech_slice, label_slice=label_slice)

        # Check whether the parallel process is done
        if self.__pipe_end_point__.poll(): # Poll returns whether the pipe stores data from the parallel process
            # Remember this information
            self.__parallel_process_is_busy__ = False
            
            # Open its message and notify user
            train_losses, validation_losses = self.__pipe_end_point__.recv() # We expect just a single message from the parallel process, containing the loss of training and validation
            self.__mean_train_loss__ = np.mean(train_losses)
            self.__mean_validation_loss__ = np.mean(validation_losses)
            self.logger.info("Train loss: " + str(np.round(self.__mean_train_loss__, decimals=2)) + " Validation loss: " + str(np.round(self.__mean_validation_loss__, decimals=2)))
        
        # If the buffer is large enough and the parallel process is ready
        if self.min_time_frames_in_buffer <= self.__time_frames_in_buffer__:# and not self.__parallel_process_is_busy__:
            # Send the buffer via the pipe
            self.__pipe_end_point__.send((self.__eeg_buffer__, self.__speech_buffer__, self.__label_buffer__))
            
            # Remember that the parallel process is busy
            self.__parallel_process_is_busy__ = True

            # Clear the buffer
            self.__reset_buffer__()

        # Set output
        time_points_in_slice = eeg_slice.size()[0]
        self.o.data = pd.DataFrame({'train': time_points_in_slice*[self.__mean_train_loss__], 'validation': time_points_in_slice*[self.__mean_validation_loss__]})
        
    def __reset_buffer__(self) -> None:
        """Mutating function that resets the buffer to empty deques.
        
        Precondition:
        - self.__eeg_buffer__, self.__speech_buffer__, and self.__label_buffer__ are non-existent attributes or collections, possibly empty.
        - self.__time_frames_in_buffer__ is a non-existent attributes or an integer.

        Inputs:
        - None

        Outputs:
        - None

        Postcondition:
        - self.__eeg_buffer__, self.__speech_buffer__, and self.__label_buffer__ are empty deques.
        - self.__time_frames_in_buffer__ is an integer equal to zero.
        """
        self.__eeg_buffer__ = collections.deque()
        self.__speech_buffer__ = collections.deque()
        self.__label_buffer__ = collections.deque()
        self.__time_frames_in_buffer__ = 0

    def __extend_buffer__(self, eeg_slice: torch.Tensor, speech_slice: torch.TensorType, label_slice: List[str]) -> None:
        """Mutating functions that enters the new data into the buffer. 
        All input slices need to have the same number of time frames for the buffer to stay valid.
        
        Precondition:
        - self.__eeg_buffer__, self.__speech_buffer__, and self.__label_buffer__ are collections, possibly empty.
        - self.__time_frames_in_buffer__ is an integer.

        Inputs:
        - eeg_slice: the x slice to be added to the buffer.
        - speech_slice: the y slice to be added to the buffer.
        - label_slice: the labels slice to be added to the buffer.
        
        Outputs:
        - None
        
        Postcondition:
        - self.__eeg_buffer__, self.__speech_buffer__, and self.__label_buffer__ got appended x_slice, y_slice and label_slice, respectively.
        - self.__time_frames_in_buffer__ got increased by the length of label_slice."""

        # Input validity
        assert type(eeg_slice) == torch.Tensor, f"Expected x_slice to have type torch.Tensor, received {type(eeg_slice)}"
        assert type(speech_slice) == torch.Tensor, f"Expected y_slice to have type torch.Tensor, received {type(speech_slice)}"
        assert type(label_slice) == type([]), f"Expected label_slice to have type List[str], received {type(label_slice)}"
        assert eeg_slice.size()[0] == speech_slice.size()[0] and speech_slice.size()[0] == len(label_slice), f"Expected x_slice, y_slice and label_slice to have the same number of time frames along the 0th axis. Received {eeg_slice.size()[0]}, {speech_slice.size()[0]} and {len(label_slice)} time frames, respectively."

        # Fill the buffers
        self.__eeg_buffer__.append(eeg_slice)
        self.__speech_buffer__.append(speech_slice)
        self.__label_buffer__.append(label_slice)

        # Update the number of time frames in the buffer
        self.__time_frames_in_buffer__ += len(label_slice)

    @staticmethod
    def parallel_fitter(pipe_end_point: Connection, neural_network_type: str, parameters_path: str, start_from_prefit: bool) -> None:
        """This function feeds the data its receives via the pipe to a models.Fitter object. It should be run on a separate process.
        It expects an initial data slice via the pipe. It will then do one fitting routine and when it is finished it will send the 
        validation and train loss via the pipe. Thereafter it is ready for the next data.
        
        Inputs:
        - pipe_end_point: A connection end point via which data is received in the form of deques of 
            eeg and spectrogram slices and a nested list of label slices. The enpoint is used to send the training and validation losses as lists of floats, respectively.
        - neural_network_type: The type of the neural network to be trained, e.g. Dense or Convolutional. This type will be taken from models.neural_networks.
        - parameters_path: The path to the file where the neural network parameters shall be stored after each fit operation. This path is relative to the parameters directory.
        - start_from_prefit: Indicates whether the neural network should load the network parameters from parameters_path before the first fit iteration. If set to False then fitting starts from random parameters.
        """

        # Create a fitter
        fitter = mf.Fitter(is_streamable=True)

        # Create variable for neural network and optimizer
        stationary_neural_network = None
        optimizer = None

        while pipe_end_point.poll(None): # Blocks process until data is available
            # Extract buffer from pipe
            eeg_buffer, speech_buffer, label_buffer = pipe_end_point.recv() # Executing awaits the return of this call
            eeg = torch.cat(list(eeg_buffer), dim=0) # Time axis
            speech = torch.cat(list(speech_buffer), dim=0) # Time axis
            labels = utilities.flatten_list(nested_list=list(label_buffer))

            # Scale
            eeg=(eeg-torch.mean(eeg,axis=1).unsqueeze(1))/torch.std(eeg,axis=1).unsqueeze(1)
            
            # Ensure neural network and optimizer exist
            if stationary_neural_network == None:
                # Shapes
                eeg_feature_count = eeg.size()[-1]; speech_feature_count = speech.size()[-1]
                
                # Get the actual type of the neural network
                neural_network_type = getattr(neural_networks, neural_network_type)
                
                # Initialize
                stationary_neural_network = neural_network_type(input_feature_count=eeg_feature_count, output_feature_count=speech_feature_count, is_streamable=False)
                
                # Optionally load parameters
                if start_from_prefit and os.path.exists(parameters_path):
                    stationary_neural_network.load_state_dict(torch.load(parameters_path))

                # Construct optimizer
                optimizer = torch.optim.Adam(params=stationary_neural_network.parameters(), lr=0.01)

            # Reshape
            eeg_reshaped, speech_reshaped = utilities.reshape_by_label(x=eeg, labels=labels, pause_string='', y=speech) # Shape == [instance count, time frame count, channel count] where each instance is one label
            instance_count = eeg_reshaped.size()[0] # Corresponds to number of words shown on screen

            # Fit
            tick = time.time()
            train_losses, validation_losses = fitter.fit(stationary_neural_network=stationary_neural_network, x=eeg_reshaped, y=speech_reshaped, loss_function=torch.nn.MSELoss(), optimizer=optimizer, instances_per_batch=(int)(0.66*instance_count), epoch_count=6, validation_proportion=0.33, is_final_slice=False, pad_axis=1)
            print(f"Fit required {time.time()-tick} seconds.")
            
            # Save the progress
            torch.save(stationary_neural_network.state_dict(), parameters_path)

            # Send the result
            pipe_end_point.send([train_losses, validation_losses])
            
    def terminate(self) -> None:
        # Wait for final fit
        if not self.__skip__ and self.__parallel_process_is_busy__:
            # Open its message and notify user
            train_losses, validation_losses = self.__pipe_end_point__.recv() # We expect just a single message from the parallel process, containing the loss of training and validation
            self.logger.info(f"Train loss: {torch.mean(train_losses)}, Validation loss: {torch.mean(validation_losses)}")

        self.__pipe_end_point__.close() # Close the connection to the parallel process
        return super().terminate()
