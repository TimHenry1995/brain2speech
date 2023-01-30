import torch
import torch.nn as nn
from abc import ABC
from sklearn.utils import shuffle
from typing import Tuple, Callable, List, Dict
import sys
sys.path.append('.')
from models import utilities
import time
# Neural Network
class Fitter():
    """This class defines a fitter for neural networks. 
    This fitter can be run in a streamable or stationary mode.
    Beware that it always fits a stationary neural network and NOT a streamable neural network."""

    def __init__(self, is_streamable: bool) -> object:
        """Constructor for this class.

        Inputs:
        - is_streamable: Indicates whether this fitter object shall fit the stationary neural network on all data at once (is_streamable=False) or sequentially using an internal buffer (is_streamable=True).
        """
        self.__is_streamable__ = is_streamable


        # Create buffers for streamable fitting
        self.__fit_x_buffer__ = []
        self.__fit_y_buffer__ = []

    def fit(self, stationary_neural_network: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, loss_function: Callable, optimizer: Callable,
          instances_per_batch: int = 8, epoch_count: int = 25, validation_proportion: float = 0.33, 
          shuffle_seed: int = 42, is_final_slice: bool = None, pad_axis: int = None) -> Tuple[List[float], List[float]]:
        """Fits the stationary_neural_network to the data. 
        
        If this fitter is in stationary mode:
        - it is equivalent to a casual model fitting where each instance is shown to the model in an arbitrary order. 
        
        If this fitter is in streamable mode:
        - the neural network will accumulate x over time where time is expected to be axis 0. If this function is called
        for the first time x will be shown once per epoch. If this function is called for the second time x will be shown twice per epoch and the previously
        provided x will be shown once per epoch. If this function is called for the kth time (k => 1), x will be shown k times per epoch and all previously provided
        xs will be shown once per epoch. This ensures that when the final slice of x is provided all slices have been shown equally often to the neural network.
        Observe that at any point in time the shape of the cost function will depend with a proportion of k/(2k-1) on the current slice and (k-1)/(2k-1) on the union of the previous slices of x. 
        For this strategy to work it is expected that the slices contain the same number of instances and that the epoch_count is constant across calls. 
        When is_final_slice == True the buffer for x will be cleared. Clearing the buffer will accelerate subsequent calls to fit, yet then it will no longer hold that
        the neural network was shown all slices equally often. The above reasoning also applies to y.
        
        Inputs:
        - stationary_neural_network: A neural network whose modules are all stationary, i.e. regular torch.nn.Modules and NOT their streamable equivalents.
        - x: the input to the neural network. Shape == [instance_count, ...]
        - y: the target output of the neural network. Shape == [instance_count, ...]
        - loss_function: a torch loss function used to determine the match of target and prediction.
        - optimizer: the torch.optim optimizer that executes the parameter update.
        - instances_per_batch: number of instances per batch.
        - epoch_count: number of iterations across training portion of the data.
        - validation_proportion: floating point number in range (0,1) indicating proportion of instances used for validation. The remaining proportion will be used for training.
        - shuffle_seed: the seed used to set the shuffle function of sklearn when shuffling instances generated from x.
        - is_final_slice: ignored in stationary mode, required in streamable mode. Indicates whether this is the final slice of the data stream.
        - pad_axis: ignored in stationary mode, optional in streamable mode. The slices that are received during fitting may need to be padded along a particular axis. With this input argument that axis can be specified. If None then no padding will be performed.

        Outputs:
        - train_lossess: Losses for train data per epoch.
        - validation_losses: Losses for validation data per epoch.
        """

        # Validate input
        assert type(x) == torch.Tensor, f"Input x must be of type torch.Tensor, not {type(x)}."
        assert type(y) == torch.Tensor, f"Input y must be of type torch.Tensor, not {type(y)}."
        assert x.size()[0] == y.size()[0], f"Inputs x and y must have same number of instances (first axis). Received x of shape {x.size()} and y of shape {y.size()}."
        if self.__is_streamable__: assert is_final_slice != None, f"Input is_final_slice is required in streamable mode. Received {is_final_slice}."

        # Handle streamable mode
        if self.__is_streamable__:
            # Insert it into the buffer
            self.__fit_x_buffer__.append(x)
            self.__fit_y_buffer__.append(y)
            
            # Optionally pad along specified axis
            if pad_axis != None:
                self.__fit_x_buffer__ = utilities.zero_pad_sequences(sequences=self.__fit_x_buffer__, axis=pad_axis)
                self.__fit_y_buffer__ = utilities.zero_pad_sequences(sequences=self.__fit_y_buffer__, axis=pad_axis)
                x = self.__fit_x_buffer__[-1]
                y = self.__fit_y_buffer__[-1]

            # Copy the current x and y k times along the instance axis
            k = len(self.__fit_x_buffer__)
            x = x.repeat(repeats=[k] + [1] * len(x.size()[1:]))
            y = y.repeat(repeats=[k] + [1] * len(y.size()[1:]))

            # Concatenate this repeated x with the other slices from the buffer
            x_new = torch.concat(self.__fit_x_buffer__[:-1] + [x], dim=0)
            y_new = torch.concat(self.__fit_y_buffer__[:-1] + [y], dim=0)

            # Rename
            x = x_new
            y = y_new

        # Shuffle x and y
        x, y = shuffle(x, y, random_state=shuffle_seed)

        # Split into train and validation portions
        split_index = (int)(validation_proportion*x.shape[0])
        train_x, validation_x = x[split_index:,:], x[:split_index,:]
        train_y, validation_y = y[split_index:,:], y[:split_index,:]

        train_losses = []
        validation_losses = []

        # Iterate epochs
        for e in range(epoch_count):
            # Enter train mode
            stationary_neural_network.train(True)

            # Validation loss
            validation_y_hat = stationary_neural_network(validation_x)
            validation_losses.append(loss_function(validation_y_hat, validation_y).detach().numpy())

            # Training
            train_losses_b = 0
            for b in range(int(len(train_x)/instances_per_batch)):
                train_x_b, train_y_b = train_x[b*instances_per_batch:(b+1)*instances_per_batch], train_y[b*instances_per_batch:(b+1)*instances_per_batch]
                
                train_y_hat = stationary_neural_network(train_x_b)
                train_loss = loss_function(train_y_hat, train_y_b)
                time.sleep(0.3)
                train_loss.backward()
                train_losses_b += train_loss.item()
                optimizer.step()
                stationary_neural_network.zero_grad()

            train_losses.append(train_losses_b / int(len(train_x)/instances_per_batch))

        # Exit train mode
        stationary_neural_network.train(False)

        # Handle streamable mode
        if self.__is_streamable__:
            if is_final_slice: 
                self.__fit_x_buffer__ = []
                self.__fit_y_buffer__ = []

        return train_losses, validation_losses 
