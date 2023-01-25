import sys
sys.path.append(".")
from models import neural_networks as mnn
from models import fitter as mft
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Type, Dict, Any

class NeuralNetwork():
    """This class provides units tests for neural_networks.NeuralNetwork"""

    @staticmethod
    def test_A(stationary_neural_network: mnn.NeuralNetwork, type: Type, instance_count: int, time_frame_count: int,
        input_feature_count: int, output_feature_count: int) -> None:
        """This unit test fits the neural network using a stationary fitter and then predicts in stationary mode.
        
        Inputs:
        - stationary_neural_network: the neural network to be fitted.
        - type: the type of the neural network.
        - instance_count: the number of instances to be used.
        - time_frame_count: the number of time frames to be used.
        - input_feature_count: the number of input features to be used.
        - output_feature_count: the number of output features to be used.
        
        Ouputs:
        - None"""
         
        # Create data
        x = torch.ones(size=(instance_count,time_frame_count,input_feature_count))
        y = torch.zeros(size=(instance_count,time_frame_count,output_feature_count))

        # Fit
        optimizer = torch.optim.Adam(params=stationary_neural_network.parameters(), lr=0.01)
        fitter = mft.Fitter(is_streamable=False)
        train_losses, validation_losses = fitter.fit(stationary_neural_network=stationary_neural_network, x=x, y=y, 
            loss_function=torch.nn.functional.mse_loss, optimizer=optimizer, 
            epoch_count=100, shuffle_seed=42)

        # Predict
        y_hat = stationary_neural_network.predict(x=x)

        # Evaluate
        is_equal = y.size() == y_hat.size()
        if is_equal:
            is_equal = torch.allclose(input=y, other=y_hat, atol=0.1)

        # Log
        print("\tPassed" if is_equal else "\tFailed", f"unit test A for {type}.")

    @staticmethod
    def test_B(stationary_neural_network: mnn.NeuralNetwork, type: str, kwargs: Dict[str, Any], instance_count: int, 
    time_frames_per_slice: int, slice_count: int, input_feature_count: int, test_name: str ='B') -> None:
        """This unit test lets the stationary_neural_network predict for an arbitrary input of shape [instance_count, time_frame_count, feature_count], 
        converts the neural network to a streamable equivalent and then predicts for the same input in streamable mode.
        It then compares whether the two models gave the same output.
        
        Inputs:
        - stationary_neural_network: the neural network to be fitted.
        - type: the type of the neural network.
        - kwargs: a dictionary with all the keyword arguments used to construct an instance of the neural network excluding the is_streamable argument.
        - instance_count: the number of instances to be used.
        - time_frames_per_slice: the number of time frames for each slice during streaming.
        - slice_count: Integer sufficiently large for the states of the streamable module to accumulate. The number of slices used during streaming.
        - input_feature_count: the number of input features to be used.
        - test_name: the name of the test as shown when printing the test result.

        Ouputs:
        - None"""
      
        # Shapes
        time_frame_count = time_frames_per_slice * slice_count
        
        # Create data
        x = torch.rand(size=(instance_count,time_frame_count,input_feature_count))
        
        # Predict stationary
        stationary_y_hat = stationary_neural_network.predict(x=x)

        # Convert to streamable
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, 'temporaryUnitTestParameters.pt')
        torch.save(stationary_neural_network.state_dict(), path)
        streamable_neural_network = type(**kwargs, is_streamable=True)
        streamable_neural_network.load_state_dict(torch.load(path))
        os.remove(path)

        # Predict stream
        streamable_y_hat = [None] * slice_count
        for i in range(slice_count):
            x_i = x[:,i*time_frames_per_slice:(i+1)*time_frames_per_slice,:] # Slice along temporal axis
            streamable_y_hat[i] = streamable_neural_network.predict(x=x_i, is_final_slice=i==slice_count-1)
        streamable_y_hat = torch.cat(streamable_y_hat, dim=1) # Concatenate along temporal axis

        # Evaluate
        is_equal = stationary_y_hat.size() == streamable_y_hat.size()
        if is_equal:
            is_equal = torch.allclose(input=stationary_y_hat, other=streamable_y_hat, atol=0.1)

        # Log
        print("\tPassed" if is_equal else "\tFailed", f"unit test {test_name} for {type}.")

class Dense():
    """This class provides unit tests for neural_networks.Dense."""

    @staticmethod
    def test_A() -> None:
        input_feature_count=3
        output_feature_count=6
        dense = mnn.Dense(input_feature_count=input_feature_count, output_feature_count=output_feature_count, is_streamable=False)
        NeuralNetwork.test_A(stationary_neural_network=dense, type=mnn.Dense, instance_count=16, time_frame_count=64, 
            input_feature_count=input_feature_count, output_feature_count=output_feature_count)

    @staticmethod
    def test_B() -> None:
        kwargs = {'input_feature_count':3, 'output_feature_count':6}
        dense = mnn.Dense(**kwargs, is_streamable=False)
        NeuralNetwork.test_B(stationary_neural_network=dense, type=mnn.Dense, kwargs=kwargs,
        instance_count=16, time_frames_per_slice = 8, slice_count=64, 
            input_feature_count=kwargs['input_feature_count'])

class Convolutional():
    """This class provides unit tests for neural_networks.Convolutional."""

    @staticmethod
    def test_A() -> None:
        input_feature_count=3
        output_feature_count=6
        convolutional = mnn.Convolutional(input_feature_count=input_feature_count, output_feature_count=output_feature_count, is_streamable=False)
        NeuralNetwork.test_A(stationary_neural_network=convolutional, type=mnn.Convolutional, instance_count=16, time_frame_count=64, 
            input_feature_count=input_feature_count, output_feature_count=output_feature_count)

    @staticmethod
    def test_B() -> None:
        kwargs = {'input_feature_count':3, 'output_feature_count':6}
        convolutional = mnn.Convolutional(**kwargs, is_streamable=False)
        NeuralNetwork.test_B(stationary_neural_network=convolutional, type=mnn.Convolutional, kwargs=kwargs, 
        instance_count=16, time_frames_per_slice = 8, slice_count=64,  
            input_feature_count=kwargs['input_feature_count'])

    @staticmethod
    def test_C() -> None:
        kwargs = {'input_feature_count':3, 'output_feature_count':6}
        convolutional = mnn.Convolutional(**kwargs, is_streamable=False)
        NeuralNetwork.test_B(stationary_neural_network=convolutional, type=mnn.Convolutional, kwargs=kwargs, 
        instance_count=16, time_frames_per_slice = 1, slice_count=64,  
            input_feature_count=kwargs['input_feature_count'], test_name='C')

if __name__ == "__main__":
    print("\nUnit tests for models.neural_networks.")

    # Dense
    Dense.test_A()
    Dense.test_B()

    # Convolutional
    Convolutional.test_A()
    Convolutional.test_B()
    Convolutional.test_C()
