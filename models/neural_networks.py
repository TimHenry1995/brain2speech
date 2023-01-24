import torch
import torch.nn as nn
from abc import ABC
from sklearn.utils import shuffle
from typing import Tuple, Callable, List, Union
import sys
sys.path.append(".")
from models import streamable_modules as streamable
from models import module_converter as mc


# Neural Network
class NeuralNetwork(ABC):
    """This abstract base class defines a set of attributes and methods for neural networks implemented in torch.
    A neural network is here defined as a graph of modules. It supports a stationary and a streamable mode which is fixed for the object's lifetime.
    It is also required that a given python process only uses a single streamable neural network at a time to prevent interference of module states."""
  
    def __init__(self, is_streamable: bool) -> object:
        """Base initializer to be inherited and extended by subclasses. 
        The subclass initializer is expected to 
        1. call this abstract base class initializer
        2. initialize a torch.nn.Module attribute self.graph which will be used for the neural networks forward and backward operations, managing parameters for saving and loading etc., e.g. a torch.Sequential. 
        This graph may only use stationary modules and they have to be convertible to streamable modules by the module_converter package.
        3. set a conditional if is_streamable: self.__to_streamable__ to ensure the graph is usable in both streamable and stationary modes. 
        
        Inputs:
        - is_streamable: Indicates whether this neural network shall be used in streaming or stationary mode.
        """
        super(NeuralNetwork, self).__init__()
        self.__is_streamable__ = is_streamable

        # Create buffers for streamable fitting
        self.__fit_x_buffer__ = []
        self.__fit_y_buffer__ = []

    def __to_streamable__(self, unravelled_trainable_modules: List[Union[torch.nn.Module, streamable.Module]]) -> None:
        """Sets the internal modules to streaming or stationary modules.  
        
        Inputs:
        - is_streamable: Indicates whether the modules or self should be set to streaming or stationary modules.
        - unravelled_trainable_modules: lists all the modules that are trainable, e.g. Linear, Conv1d and dot not contain submodules. If the inheriting subclass uses for instance a GRU 
        then the submodules of the GRU have to be listed one by one here.
        
        Outputs:
        - None"""

        # Replace the __call__ or forward method of self with its alternative
        for m, module in enumerate(unravelled_trainable_modules):

            # Map from stationary to streamable
            unravelled_trainable_modules[m] = mc.ModuleConverter.stationary_to_streamable(module=module)
            
    def fit(self, x: torch.Tensor, y: torch.Tensor, loss_function: Callable, optimizer: Callable,
          instances_per_batch: int = 8, epoch_count: int = 25, validation_proportion: float = 0.3, 
          shuffle_seed: int = 42, is_final_slice: bool = None) -> Tuple[List[float], List[float]]:
        """Fits the model to the data. When in stationary mode this is equivalent to a casual model fitting where each instance is 
        shown to the model in an arbitrary order. When in streamable mode the neural network will accumulate x over time. If this function is called
        for the first time x will be shown once per epoch. If this function is called for the second time x will be shown twice per epoch and the previously
        provided x will be shown once per epoch. If this function is called for the kth time (k => 1), x will be shown k times per epoch and all previously provided
        xs will be shown once per epoch. This ensures that when the final slice of x is provided all slices have been shown equally often to the neural network.
        Observe that at any point in time the shape of the cost function will depend with a proportion of k/(2k-1) on the current slice and (k-1)/(2k-1) on the union of the previous slices of x. 
        For this strategy to work it is expected that the slices contain the same number of instances and that the epoch_count is constant across calls. 
        When is_final_slice == True the buffer for x will be cleared. Clearing the buffer will accelerate subsequent calls to fit, yet then it will no longer hold that
        the neural network was shown all slices equally often. The above reasoning also applies to y.
        
        Inputs:
        - x: the input to the neural network. Shape == [instance_count, time frame count, input feature count]
        - y: the target output of the neural network. Shape == [instance_count, time frame count, output feature count]
        - loss_function: a torch loss function used to determine the match of target and prediction.
        - optimizer: the torch.optim optimizer that executes the parameter update.
        - instances_per_batch: number of instances per batch.
        - epoch_count: number of iterations across training portion of the data.
        - validation_proportion: floating point number in range (0,1) indicating proportion of instances used for validation. The remaining proportion will be used for training.
        - shuffle_seed: the seed used to set the shuffle function of sklearn when shuffling instances generated from x.
        - is_final_slice: optional in stationary mode, required in streamable mode. Indicates whether this is the final slice of the data stream.

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
            # Copy the current x and y k times along the instance axis
            k = len(self.__fit_x_buffer__) + 1
            x = x.repeat(repeats=[k] + [1] * len(x.size()[1:]))
            y = y.repeat(repeats=[k] + list(y.size()[1:]))

            # Concatenate with the buffer
            x_new = torch.concat(self.__fit_x_buffer__ + [x], dim=0)
            y_new = torch.concat(self.__fit_y_buffer__ + [y], dim=0)

            # Insert it into the buffer
            self.__fit_x_buffer__.append(x)
            self.__fit_y_buffer__.append(y)

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
            self.graph.train(True)

            # Validation loss
            validation_y_hat = self.graph(validation_x)
            validation_losses.append(loss_function(validation_y_hat, validation_y).detach().numpy())

            # Training
            train_losses_b = 0
            for b in range(int(len(train_x)/instances_per_batch)):
                train_x_b, train_y_b = train_x[b*instances_per_batch:(b+1)*instances_per_batch], train_y[b*instances_per_batch:(b+1)*instances_per_batch]
                
                train_y_hat = self.graph(train_x_b)
                train_loss = loss_function(train_y_hat, train_y_b)
                
                train_loss.backward()
                train_losses_b += train_loss.item()
                optimizer.step()
                self.graph.zero_grad()

            train_losses.append(train_losses_b / int(len(train_x)/instances_per_batch))

        # Exit train mode
        self.graph.train(False)

        # Handle streamable mode
        if self.__is_streamable__:
            if is_final_slice: 
                self.__fit_x_buffer__ = []
                self.__fit_y_buffer__ = []

        return train_losses, validation_losses 
  
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
        y_hat = self.graph(x)

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
    
        # Super
        super(Dense, self).__init__(is_streamable=is_streamable)
  
        # Create graph
        linear_1 = nn.Linear(input_feature_count, output_feature_count)
        linear_2 = nn.Linear(output_feature_count, output_feature_count)
        self.graph = nn.Sequential(linear_1, nn.ReLU(), linear_2)
  
        # Set streamable
        if is_streamable: self.__to_streamable__(unravelled_trainable_modules=[linear_1, linear_2])
