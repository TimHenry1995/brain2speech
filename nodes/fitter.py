from timeflux.core.node import Node
import matplotlib.pyplot as plt, numpy as np
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
    """This node manages the flow of incoming data to a models.fitter.Fitter object which runs on a parallel process."""

    def __init__(self, min_time_frames_in_buffer: int, eeg_stream_name: str, speech_stream_name: str, label_stream_name: str, neural_network_type: str, parameters_path: str) -> object:
        """Constructor for this class.
        
        Inputs:
        - min_time_frames_in_buffer: The minimum number of time frames 
            that have to be in the buffer in order to send them to the models.Fitter object for fitting.
        - eeg_stream_name, speech_stream_name, label_stream_name: The names of the streams. Each such name is used to identify a stream from the unifier.
        - neural_network_type: The type of the neural network to be trained, e.g. Dense or Convolutional. This type will be taken from models.neural_networks.
        - parameters_path: The path to the file where the neural network parameters shall be stored after each fit operation. This path is relative to the parameters directory.
        """

        # Copy attributes
        self.min_time_frames_in_buffer = min_time_frames_in_buffer
        self.eeg_stream_name = eeg_stream_name
        self.speech_stream_name = speech_stream_name
        self.label_stream_name = label_stream_name
        
        # Set the buffer
        self.__reset_buffer__()

        # Set up parallel process
        self.__pipe_end_point__, other_end_point = Pipe()
        self.__parallel_process__ = Process(target=self.parallel_fitter, args=([other_end_point, neural_network_type, parameters_path]))
        self.__parallel_process__.daemon = True # Ensures that the parallel process is killed when the main process is killed.
        self.__parallel_process__.start()
        self.__parallel_process_is_busy__ = False # The other process is only considered busy when it is currently processing data

    def update(self) -> None:
        """Maintains a buffer of EEG, spectrogram and labels and submits it regularly to a models.fitter.Fitter object. 
        Expects to receive input in the form of a dictionary with keys for the eeg, speech and label streams and values for 
        their respective data frames. As data streams into this node it will be stored in a buffer. 
        Once the number of received time frames surpasses self.min_time_frames_in_buffer the buffer is sent to a models.fitter.Fitter object which
        will fit a model on them on a parallel process. While the models.fitter.Fitter is busy this update function will buffer the new slices of
        incoming data. As soon as the buffer surpasses self.min_time_frames_in_buffer the fitter node will take
        the next opportunity to send it to the models.fitter.Fitter and the cycle continues."""
        
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
            self.logger.info("Train loss: " + str(np.round(np.mean(train_losses), decimals=2)) + " Validation loss: " + str(np.round(np.mean(validation_losses), decimals=2)))
        
        # If the buffer is large enough and the parallel process is ready
        if self.min_time_frames_in_buffer <= self.__time_frames_in_buffer__:# and not self.__parallel_process_is_busy__:
            # Send the buffer via the pipe
            self.__pipe_end_point__.send((self.__eeg_buffer__, self.__speech_buffer__, self.__label_buffer__))
            
            # Remember that the parallel process is busy
            self.__parallel_process_is_busy__ = True

            # Clear the buffer
            self.__reset_buffer__()

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
    def parallel_fitter(pipe_end_point: Connection, neural_network_type: str, parameters_path: str) -> None:
        """This function feeds the data its receives via the pipe to a models.Fitter object. It should be run on a separate process.
        It expects an initial data slice via the pipe. It will then do one fitting routine and when it is finished it will send the 
        validation and train loss via the pipe. Thereafter it is ready for the next data.
        
        Inputs:
        - pipe_end_point: A connection end point via which data is received in the form of deques of 
            eeg and spectrogram slices and a nested list of label slices. The enpoint is used to send the training and validation losses as lists of floats, respectively.
        - neural_network_type: The type of the neural network to be trained, e.g. Dense or Convolutional. This type will be taken from models.neural_networks.
        - parameters_path: The path to the file where the neural network parameters shall be stored after each fit operation. This path is relative to the parameters directory.
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
            
            # Reshape
            eeg, speech = utilities.reshape_by_label(x=eeg, labels=labels, pause_string='', y=speech) # Shape == [instance count, time frame count, channel count] where each instance is one label
            instance_count = eeg.size()[0] # Corresponds to number of words shown on screen

            # Ensure neural network and optimizer exist
            if stationary_neural_network == None:
                # Shapes
                eeg_feature_count = eeg.size()[-1]; speech_feature_count = speech.size()[-1]
                
                # Get the actual type of the neural network
                neural_network_type = getattr(neural_networks, neural_network_type)
                
                # Initialize
                stationary_neural_network = neural_network_type(input_feature_count=eeg_feature_count, output_feature_count=speech_feature_count, is_streamable=False)
                optimizer = torch.optim.Adam(params=stationary_neural_network.parameters(), lr=0.01)

            # Fit
            train_losses, validation_losses = fitter.fit(stationary_neural_network=stationary_neural_network, x=eeg, y=speech, loss_function=torch.nn.MSELoss(), optimizer=optimizer, instances_per_batch=(int)(0.66*instance_count), epoch_count=5, validation_proportion=0.33, is_final_slice=False, pad_axis=1)
            
            # Save the progress
            torch.save(stationary_neural_network.state_dict(), os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'parameters', parameters_path + '.pt')))

            # Send the result
            pipe_end_point.send([train_losses, validation_losses])
            
    def terminate(self) -> None:
        # Wait for final fit
        if self.__parallel_process_is_busy__:
            # Open its message and notify user
            train_losses, validation_losses = self.__pipe_end_point__.recv() # We expect just a single message from the parallel process, containing the loss of training and validation
            self.logger.info(f"Train loss: {torch.mean(train_losses)}, Validation loss: {torch.mean(validation_losses)}")

        self.__pipe_end_point__.close() # Close the connection to the parallel process
        return super().terminate()

    def plot_x_target_and_output(self, x: torch.Tensor, target: torch.Tensor, output: torch.Tensor, labels: List[str], pause_string: str, path: str) -> None:
        """Plots x, the target and output.
        Inputs:
        - x: Input EEG time series. Shape == [time frame count, eeg channel count].
        - target: Desired spectrogram. Shape == [time frame count, mel channel count].
        - output: Obtained spectrogram. Shape == [time frame count, mel channel count].
        - labels: The labels that indicate for each time frame of x, target and output which label was present at that time. Length == time frame count.
        - pause_string: The string used to indicate pauses.
        - path: Path to the folder where the figure should be stored.

        Assumptions:
        - x, target, output, labels are expected to have the same time frame count.
        - target, output are expected to have the same shape.

        Outputs:
        - None
        """
        # Input validity
        assert type(x) == torch.Tensor, f"Expected x to have type torch.Tensor, received {type(x)}."
        assert type(target) == torch.Tensor, f"Expected target to have type torch.Tensor, received {type(target)}."
        assert type(output) == torch.Tensor, f"Expected target to have type torch.Tensor, received {type(target)}."
        assert type(labels) == type(['']), f"Expected labels to have type {type([''])}, received {type(labels)}."
        assert x.size()[0] == output.size()[0] and output.size()[0] == target.size()[0] and target.size()[0] == len(labels), f"Expected x, target, output and labels to have the same time frame count. Received for x {x.size()[0]}, target {target.size()[0]}, output {output.size()[0]}, labels {len(labels)}."

        # Figure
        fig=plt.figure()
        plt.suptitle("Sample of Data Passed Through " + self.model_name)

        # Labels
        tick_locations = [0]
        tick_labels = [labels[0]]
        for l in range(1,len(labels)):
            if labels[l] != labels[l-1] and labels[l] != pause_string: 
                tick_locations.append(l)
                tick_labels.append(labels[l]) 

        # EEG
        plt.subplot(3,1,1); plt.title("EEG Input")
        plt.imshow(x.permute((1,0)).detach().numpy()); plt.ylabel("EEG Channel")
        plt.xticks(ticks=tick_locations, labels=['' for label in tick_labels])

        # Target spectrogram
        plt.subplot(3,1,2); plt.title("Target Speech Spectrogram")
        plt.imshow(np.flipud(target.permute((1,0)).detach().numpy()))
        plt.xticks(ticks=tick_locations, labels=['' for label in tick_labels])
        plt.ylabel("Mel Channel")
        
        # Output spectrogram
        plt.subplot(3,1,3); plt.title("Output Spech Spectrogram")
        plt.imshow(np.flipud(output.permute((1,0)).detach().numpy()))
        plt.xlabel("Time Frames")
        plt.ylabel("Mel Channel")
        plt.xticks(ticks=tick_locations, labels=tick_labels)

        # Saving
        if not os.path.exists(path): os.makedirs(path)
        plt.savefig(os.path.join(path, "Sample Data.png"), dpi=600)
        plt.close(fig)
   
    def plot_loss_trajectory(self, train_losses: List[float], validation_losses: List[float], path: str, 
                          loss_name: str, logarithmic: bool = True) -> None:
        """Plots the losses of train and validation time courses per epoch on a logarithmic scale.
        
        Assumptions:
        - train and validation losses are assumed to have the same number of elements and that their indices are synchronized.

        Inputs:
        - train_losses: The losses of the model during training.
        - validation_losses: The losses of the model during validation.
        - path: Path to the folder where the figure should be stored.
        - loss_name: Name of the loss function.
        - logarithmic: Inidcates whether the plot should use a logarithmic y-axis.
        
        Outputs:
        - None"""
    
        # Figure
        fig=plt.figure()
        
        # Transform
        if logarithmic:
            train_losses = np.log(train_losses)
            validation_losses = np.log(validation_losses)
            plt.yscale('log')

        # Plot
        plt.plot(train_losses)
        plt.plot(validation_losses)
        plt.legend(["Train","Validation"])
        plt.title("Learning curve for " + self.model_name)
        plt.xlabel("Epoch"); plt.ylabel(loss_name)
    
        # Save
        if not os.path.exists(path): os.makedirs(path)
        plt.savefig(os.path.join(path, "Learning Curve.png"), dpi=600)
        plt.close(fig=fig)
