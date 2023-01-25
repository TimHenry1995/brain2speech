from timeflux.core.node import Node
import matplotlib.pyplot as plt, numpy as np
from typing import List
import torch, os, sys
from typing import List
sys.path.append(".")
from models import fitter, neural_networks, utilities
# importing module
import collections
from multiprocessing import Process, Pipe

class Fitter(Node):
    """This node manages the flow of incoming data to a models.fitter.Fitter object which runs on a parallel process."""

    def __init__(self, min_unique_label_count_buffer: int) -> object:
        """Constructor for this class.
        
        Inputs:
        - min_unique_label_count_buffer: The minimum number of unique labels excluding the pause character '' 
            that have to be in the buffer in order for it to flush them to the models.Fitter object for fitting.
        """

        # Set attributes
        self.min_unique_label_count_buffer = min_unique_label_count_buffer
        self.__reset_buffer__()

        # Set up parallel process
        self.__pipe_end_point__, other_end_point = Pipe()
        self.__parallel_process__ = Process(target=self.parallel_fitter, args=(other_end_point))
        self.__parallel_process__.start()
        self.__parallel_process_is_busy__ = False

    def update(self) -> None:
        """Maintains a buffer of EEG, spectrogram and labels and submits it regularly to a models.fitter.Fitter object. 
        That is, as data streams into this node in form of EEG features, speech spectrogram features and labels,
        it will store them in a buffer. It uses the label stream to count how many completed
        labels it received. Once that number surpasses self.min_unique_label_count_buffer the buffer is flushed to a models.fitter.Fitter object which
        will fit a model on them. While the models.fitter.Fitter is busy this update function will buffer the new slices of
        incoming data. As soon as the buffer surpasses self.min_unique_label_count_buffer the fitter node will take
        the next opportunity to flush it to the models.fitter.Fitter and the cycle continues. 
        Here, opportunity means that the models.fitter.Fitter is ready for the next fit."""
        
        # Copy meta
        self.o.meta = self.i.meta

        # Take out x, y, labels
        x_slice, y_slice, labels_slice = self.i.data()

        # Extend the buffer by the input
        if len(labels_slice) > 0: self.__extend_buffer__(x_slice=torch.Tensor(x_slice), y_slice=torch.Tensor(y_slice), labels_slice=labels_slice)

        # Check whether the parallel process is done
        if self.__pipe_end_point__.poll(): # Poll returns whether the pipe stores data from the parallel process
            # Remember this information
            self.__parallel_process_is_busy__ = False

            # Open its message and notify user
            train_losses, validation_losses = self.__pipe_end_point__.recv() # We expect just a single message from the parallel process, containing the loss of training and validation
            self.logger.info(f"Train loss: {torch.mean(train_losses)}, Validation loss: {torch.mean(validation_losses)}")

        # If the buffer is large enough and the parallel process is ready
        if self.min_unique_label_count_buffer <= self.__unique_labels_buffer__ and not self.__parallel_process_is_busy__:
            # Send the buffer via the pipe
            self.__pipe_end_point__.send((self.__x_buffer__, self.__y_buffer__, self.__labels_buffer__))

            # Clear the buffer
            self.__reset_buffer__()

            # Remember that the parallel process is busy
            self.__parallel_process_is_busy__ = True

    def __reset_buffer__(self) -> None:
        """Resets the buffer to empty deques."""
        self.__x_buffer__ = collections.deque()
        self.__y_buffer__ = collections.deque()
        self.__labels_buffer__ = collections.deque()
        self.__unique_labels_buffer__ = collections.deque()

    def __extend_buffer__(self, x_slice: torch.Tensor, y_slice: torch.TensorType, labels_slice: List[str]) -> None:
        """Enter the new data into the buffer. All input slices need to have the same number of time points for the buffer to stay valid.
        Assumes the slices contain at least 1 time frame.

        Inputs:
        - x_slice: the x slice.
        - y_slice: the y slice.
        - labels_slice: the labels slice
        
        Outputs:
        - None"""

        # Input validity
        assert type(x_slice) == torch.Tensor, f"Expected x_slice to have type torch.Tensor, received {type(x_slice)}"
        assert type(y_slice) == torch.Tensor, f"Expected y_slice to have type torch.Tensor, received {type(y_slice)}"
        assert type(labels_slice) == type([]), f"Expected labels_slice to have type List[str], received {type(labels_slice)}"
        assert x_slice.size()[0] == y_slice.size()[0] and y_slice.size()[0] == len(labels_slice), f"Expected x_slice, y_slice and labels_slice to have the same number of time frames along the 0th axis. Received {x_slice.size()[0]}, {y_slice.size()[0]} and {len(labels_slice)} time frames, respectively."

        # Fill the buffers
        self.__x_buffer__.append(x_slice)
        self.__y_buffer__.append(y_slice)
        self.__labels_buffer__.append(labels_slice)

        # Update the label count
        new_unique_labels = set(labels_slice)
        if '' in new_unique_labels: new_unique_labels.remove('') # The empty character is assumed to separate labels
        self.__unique_labels_buffer__ += len(new_unique_labels)


    @staticmethod
    def parallel_fitter(pipe_end_point) -> None:
        """This function feeds the data its receives via the pipe to a models.Fitter object. It should be run on a separate process.
        It expects an initial data slice via the pipe. It will then do one fitting routine and when it is finished it will send the 
        validation and train loss via the pipe. Thereafter it is ready for the next data."""

        # Create a model
        stationary_neural_network = neural_networks.Dense(input_feature_count=127, output_feature_count=80, is_streamable=False)

        # Create a fitter
        fitter = fitter.Fitter(is_streamable=True)

        # Create an optimizer
        optimizer = torch.optim.Adam(params=stationary_neural_network.parameters(), lr=0.01)

        while True:
            # Take a slice
            x_buffer, y_buffer, labels_buffer = pipe_end_point.recv()
            x = torch.cat(list(x_buffer), dim=0) # Time axis
            y = torch.cat(list(y_buffer), dim=0) # Time axis
            labels = list(labels_buffer)

            # Reshape
            x, y = utilities.reshape_by_label(x=x, labels=labels, pause_string='', y=y)

            # Fit
            train_losses, validation_losses = fitter.fit(stationary_neural_network=stationary_neural_network, x=x, y=y, loss_function=torch.nn.MSELoss(), 
                optimizer=optimizer, instances_per_batch = min(8, x.size()[0]//2), epoch_count = 5, is_final_slice=False)

            # Send the result
            pipe_end_point.send([train_losses, validation_losses])

    def terminate(self) -> None:
        self.__pipe_end_point__.close() # Close the connection to the parallel process
        if self.__parallel_process_is_busy__:
            self.__pipe_end_point__.recv() # Wait for the other process to finish
        self.__parallel_process__.join() # Close the parallel process
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
