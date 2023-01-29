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
        # Set parameters since not all parameters converge equally well
        torch.seed = 42
        new_state_dict = {}
        for (name, parameter) in convolutional.state_dict().items():
            if not parameter is None:
                new_state_dict[name] = 0.1*torch.rand_like(parameter)
        convolutional.load_state_dict(new_state_dict)
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

class Attention():
    """This class provides units tests for the AttentionNeuralNetwork."""

    @staticmethod
    def test_A() -> None:
        input_feature_count=3
        output_feature_count=6

        x_key = torch.cat([torch.rand(20,input_feature_count), torch.ones([10, input_feature_count])])
        x_value = torch.cat([torch.rand(20,output_feature_count), torch.zeros([10, output_feature_count])])
        labels = ['a'] * 10 + [''] * 10 + ['b'] * 5 + [''] * 5
        attention = mnn.Attention(query_feature_count=input_feature_count, hidden_feature_count=input_feature_count, x_key=x_key, x_value=x_value, labels=labels, pause_string='', step_count = 2, step_size = 1, is_streamable= False)
        NeuralNetwork.test_A(stationary_neural_network=attention, type=mnn.Attention, instance_count=16, time_frame_count=64, 
            input_feature_count=input_feature_count, output_feature_count=output_feature_count)

    @staticmethod
    def test_B() -> None:
        input_feature_count=3
        output_feature_count=6
        
        x_key = torch.cat([torch.rand(20,input_feature_count), torch.ones([10, input_feature_count])])
        x_value = torch.cat([torch.rand(20,output_feature_count), torch.zeros([10, output_feature_count])])
        labels = ['a'] * 10 + [''] * 10 + ['b'] * 5 + [''] * 5
        kwargs = {'query_feature_count':input_feature_count, 'hidden_feature_count':input_feature_count, 'x_key':x_key, 'x_value':x_value, 'labels':labels, 'pause_string':'', 'step_count': 2, 'step_size': 1}
        
        attention = mnn.Attention(**kwargs, is_streamable=False)
        NeuralNetwork.test_B(stationary_neural_network=attention, type=mnn.Attention, kwargs=kwargs, 
        instance_count=16, time_frames_per_slice = 8, slice_count=64,  
            input_feature_count=input_feature_count)

    @staticmethod
    def test_C() -> None:
        input_feature_count=3
        output_feature_count=6
        
        x_key = torch.cat([torch.rand(20,input_feature_count), torch.ones([10, input_feature_count])])
        x_value = torch.cat([torch.rand(20,output_feature_count), torch.zeros([10, output_feature_count])])
        labels = ['a'] * 10 + [''] * 10 + ['b'] * 5 + [''] * 5
        kwargs = {'query_feature_count':input_feature_count, 'hidden_feature_count':input_feature_count, 'x_key':x_key, 'x_value':x_value, 'labels':labels, 'pause_string':'', 'step_count': 2, 'step_size': 1}
        
        attention = mnn.Attention(**kwargs, is_streamable=False)
        NeuralNetwork.test_B(stationary_neural_network=attention, type=mnn.Attention, kwargs=kwargs, 
        instance_count=16, time_frames_per_slice = 1, slice_count=64,  
            input_feature_count=input_feature_count, test_name='C')

    @staticmethod
    def test_stack_attention_A():
        x = torch.arange(start=0, end=70, step=1).reshape((1,10,7))
        y = torch.Tensor(
            [[[[ 0.,  0.,  0.,  0.,  0.,  0.,  0.],[ 0.,  0.,  0.,  0.,  0.,  0.,  0.],[ 0.,  0.,  0.,  0.,  0.,  0.,  0.],[ 0.,  0.,  0.,  0.,  0.,  0.,  0.],[ 0.,  0.,  0.,  0.,  0.,  1.,  2.],[ 0.,  0.,  0.,  0.,  7.,  8.,  9.],[ 0.,  0.,  0.,  0., 14., 15., 16.],[ 0.,  0.,  0.,  0., 21., 22., 23.],[ 0.,  0.,  0.,  0., 28., 29., 30.],[ 0.,  0.,  0.,  0., 35., 36., 37.]],
            [[ 0.,  0.,  0.,  0.,  0.,  0.,  0.],[ 0.,  0.,  0.,  0.,  0.,  0.,  0.],[ 0.,  0.,  0.,  1.,  2.,  3.,  4.],[ 0.,  0.,  7.,  8.,  9., 10., 11.],[ 0.,  0., 14., 15., 16., 17., 18.],[ 0.,  0., 21., 22., 23., 24., 25.],[ 0.,  0., 28., 29., 30., 31., 32.],[ 0.,  0., 35., 36., 37., 38., 39.],[ 0.,  0., 42., 43., 44., 45., 46.],[ 0.,  0., 49., 50., 51., 52., 53.]],
            [[ 0.,  1.,  2.,  3.,  4.,  5.,  6.],[ 7.,  8.,  9., 10., 11., 12., 13.],[14., 15., 16., 17., 18., 19., 20.],[21., 22., 23., 24., 25., 26., 27.],[28., 29., 30., 31., 32., 33., 34.],[35., 36., 37., 38., 39., 40., 41.],[42., 43., 44., 45., 46., 47., 48.],[49., 50., 51., 52., 53., 54., 55.],[56., 57., 58., 59., 60., 61., 62.],[63., 64., 65., 66., 67., 68., 69.]]]]
            )
        y_hat = mnn.Attention.__stack_attention__(attention=x, step_count=3, step_size=2)
        
        if torch.equal(y, y_hat): print("\tPassed unit test A for Attention.__shift_x_value__().")
        else: print("\tFailed unit test A for Attention.__shift_x_value__().")

    @staticmethod
    def test_stack_attention_B():
        x = torch.arange(start=0, end=70, step=1).reshape((1,10,7))
        y = torch.Tensor(
            [[[[ 0.,  1.,  2.,  3.,  4.,  5.,  6.],[ 7.,  8.,  9., 10., 11., 12., 13.],[14., 15., 16., 17., 18., 19., 20.],[21., 22., 23., 24., 25., 26., 27.],[28., 29., 30., 31., 32., 33., 34.],[35., 36., 37., 38., 39., 40., 41.],[42., 43., 44., 45., 46., 47., 48.],[49., 50., 51., 52., 53., 54., 55.],[56., 57., 58., 59., 60., 61., 62.],[63., 64., 65., 66., 67., 68., 69.]]]]
            )
        y_hat = mnn.Attention.__stack_attention__(attention=x, step_count=1, step_size=2)
        
        if torch.equal(y, y_hat): print("\tPassed unit test B for Attention.__shift_x_value__().")
        else: print("\tFailed unit test B for Attention.__shift_x_value__().")

    @staticmethod
    def test_stack_attention_C():
        x = torch.arange(start=0, end=90, step=1).reshape((3,5,6))
        y = torch.Tensor(
            [[[[ 0.,  0.,  0.,  0.,  0.,  0.], [ 0.,  0.,  0.,  0.,  0.,  0.], [ 0.,  0.,  0.,  0.,  0.,  0.], [ 0.,  0.,  0.,  0.,  1.,  2.], [ 0.,  0.,  0.,  6.,  7.,  8.]],
            [[ 0.,  1.,  2.,  3.,  4.,  5.], [ 6.,  7.,  8.,  9., 10., 11.], [12., 13., 14., 15., 16., 17.], [18., 19., 20., 21., 22., 23.], [24., 25., 26., 27., 28., 29.]]],
            [[[ 0.,  0.,  0.,  0.,  0.,  0.], [ 0.,  0.,  0.,  0.,  0.,  0.], [ 0.,  0.,  0.,  0.,  0.,  0.], [ 0.,  0.,  0., 30., 31., 32.], [ 0.,  0.,  0., 36., 37., 38.]], 
            [[30., 31., 32., 33., 34., 35.], [36., 37., 38., 39., 40., 41.], [42., 43., 44., 45., 46., 47.], [48., 49., 50., 51., 52., 53.], [54., 55., 56., 57., 58., 59.]]],
            [[[ 0.,  0.,  0.,  0.,  0.,  0.], [ 0.,  0.,  0.,  0.,  0.,  0.], [ 0.,  0.,  0.,  0.,  0.,  0.], [ 0.,  0.,  0., 60., 61., 62.], [ 0.,  0.,  0., 66., 67., 68.]],        [[60., 61., 62., 63., 64., 65.], [66., 67., 68., 69., 70., 71.], [72., 73., 74., 75., 76., 77.], [78., 79., 80., 81., 82., 83.], [84., 85., 86., 87., 88., 89.]]]]
            )
        y_hat = mnn.Attention.__stack_attention__(attention=x, step_count=2, step_size=3)
        
        if torch.equal(y, y_hat): print("\tPassed unit test C for Attention.__shift_x_value__().")
        else: print("\tFailed unit test C for Attention.__shift_x_value__().")

if __name__ == "__main__":
    print("\nUnit tests for models.neural_networks.")

    # Dense
    Dense.test_A()
    Dense.test_B()

    # Convolutional
    Convolutional.test_A()
    Convolutional.test_B()
    Convolutional.test_C()

    # Attention
    Attention.test_A()
    Attention.test_B()
    Attention.test_C()
    Attention.test_stack_attention_A()
    Attention.test_stack_attention_B()
    Attention.test_stack_attention_C()