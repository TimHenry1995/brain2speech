import sys
sys.path.append(".")
from models import streamable_modules as streamable
import torch
import numpy as np
import matplotlib.pyplot as plt

# Indicates whether tests should plot their results
plot = False

class TensorTestUtils:

    def evaluate_prediction(streamable_model: streamable.Module, stationary_model: torch.nn.Module, x: torch.Tensor, slice_size: int) -> bool:
        """Predicts the output for the streamable model and its stationary equivalent.
        
        Inputs:
        - streamable_model: The model used to covnert x to y during streamable
        - stationary_forward: The stationary model to convert x to y (no streamable).
        - x: The input data.
        - slice_size: The number of time points per slice during streamable
        
        Outputs:
        - is_equal: Indicated whether the outputs of the streamable and stationary models are equal."""
        
        # Predict stationary
        stationary_output = stationary_model(input=x)
        
        # Predict stream
        temporal_dim = 2
        time_point_count = x.size()[temporal_dim]
        streamable_outputs = []
        i = 0
        while i < time_point_count:
            x_i = x[:,:,i:i+slice_size]
            
            is_end_of_stream = i + slice_size >= time_point_count
            if is_end_of_stream:
                streamable.Module.final_slice()
            streamable_outputs.append(streamable_model(input=x_i))
            if is_end_of_stream:
                streamable.Module.close_stream()

            i += slice_size

        streamable_output = torch.cat(streamable_outputs, dim=2)
        
        # Evaluate
        is_equal = np.allclose(a=stationary_output.detach().numpy(), b=streamable_output.detach().numpy(), atol=1e-06)
        
        if plot:
            plt.figure(); 
            plt.subplot(3,1,1)
            plt.plot(stationary_output[0,0,:].detach().numpy()); plt.title("stationary output"); plt.ylabel("Amplitude"); plt.xticks([])
            
            plt.subplot(3,1,2)
            plt.plot(streamable_output[0,0,:].detach().numpy()); plt.title("streamable output"); plt.ylabel("Amplitude"); plt.xticks([])
            
            plt.subplot(3,1,3)
            plt.plot((stationary_output[0,0,:] - streamable_output[0,0,:]).detach().numpy()); plt.title("stationary - streamable Output"); plt.ylabel("Amplitude"); plt.xlabel("Time")
            tmp = np.cumsum([max(0,y_i.shape[-1]) for y_i in streamable_outputs]) -1
            tmp[tmp < 0] = 0
            plt.scatter([0] + list(tmp),[0]*(1+len(streamable_outputs)), c='r'); plt.legend(["Cuts"])
            
            plt.show()

        # Output
        return is_equal

class Conv1d():
    """Defines unit tests for the streamable_Conv1d class. Tests sample from combinations of stride, dilation, padding, kernel size and slice size."""

    def forward_A():
        """Tests the forward method of this class."""

        # Create example data
        time_point_count = 241
        instance_count = 16
        in_channels = 32
        x = torch.randn(instance_count, in_channels, time_point_count)

        # Create streamable model
        out_channels = 17
        kernel_size = 11
        dilation = 1
        stride = 5
        padding = 0
        slice_size = 3
        streamable_model = streamable.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=0)
        
        # Create stationary model
        stationary_model = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=0)
        stationary_model.weight = streamable_model.weight
        stationary_model.bias = streamable_model.bias

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streamable_model=streamable_model, stationary_model=stationary_model, x=x, slice_size=slice_size)
        print("\tPassed" if is_equal else "\tFailed", "unit test A for Conv1d.")

    def forward_B():
        """Tests the forward method of this class."""

        # Create example data
        time_point_count = 111
        instance_count = 16
        in_channels = 32
        x = torch.randn(instance_count, in_channels, time_point_count)

        # Create streamable model
        out_channels = 17
        kernel_size = 14
        dilation = 3
        stride = 7
        padding = 3
        slice_size = 27
        streamable_model = streamable.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        
        # Create stationary model
        stationary_model = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        stationary_model.weight = streamable_model.weight
        stationary_model.bias = streamable_model.bias

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streamable_model=streamable_model, stationary_model=stationary_model, x=x, slice_size=slice_size)
        print("\tPassed" if is_equal else "\tFailed", "unit test B for Conv1d.")

    def forward_C():
        """Tests the forward method of this class."""

        # Create example data
        time_point_count = 41
        instance_count = 16
        in_channels = 32
        x = torch.randn(instance_count, in_channels, time_point_count)

        # Create streamable model
        out_channels = 17
        kernel_size = 1
        dilation = 3
        stride = 5
        padding = 0
        slice_size = 13
        streamable_model = streamable.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        
        # Create stationary model
        stationary_model = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        stationary_model.weight = streamable_model.weight
        stationary_model.bias = streamable_model.bias

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streamable_model=streamable_model, stationary_model=stationary_model, x=x, slice_size=slice_size)
        print("\tPassed" if is_equal else "\tFailed", "unit test C for Conv1d.")

    def forward_D():
        """Tests the forward method of this class."""

        # Create example data
        time_point_count = 41
        instance_count = 16
        in_channels = 32
        x = torch.randn(instance_count, in_channels, time_point_count)

        # Create streamable model
        out_channels = 17
        kernel_size = 1
        dilation = 3
        stride = 5
        padding = 11
        slice_size = 11
        streamable_model = streamable.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        
        # Create stationary model
        stationary_model = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        stationary_model.weight = streamable_model.weight
        stationary_model.bias = streamable_model.bias

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streamable_model=streamable_model, stationary_model=stationary_model, x=x, slice_size=slice_size)
        print("\tPassed" if is_equal else "\tFailed", "unit test D for Conv1d.")

    def forward_E():
        """Tests the forward method of this class."""

        # Create example data
        time_point_count = 48
        instance_count = 16
        in_channels = 32
        x = torch.randn(instance_count, in_channels, time_point_count)

        # Create streamable model
        out_channels = 17
        kernel_size = 5
        dilation = 4
        stride = 2
        padding = 11
        slice_size = 3
        streamable_model = streamable.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        
        # Create stationary model
        stationary_model = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        stationary_model.weight = streamable_model.weight
        stationary_model.bias = streamable_model.bias

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streamable_model=streamable_model, stationary_model=stationary_model, x=x, slice_size=slice_size)
        print("\tPassed" if is_equal else "\tFailed", "unit test E for Conv1d.")

    def forward_F():
        """Tests the forward method of this class."""

        # Create example data
        time_point_count = 48
        instance_count = 16
        in_channels = 32
        x = torch.randn(instance_count, in_channels, time_point_count)

        # Create streamable model
        out_channels = 17
        kernel_size = 7
        dilation = 1
        stride = 1
        padding = 0
        slice_size = 3
        streamable_model = streamable.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        
        # Create stationary model
        stationary_model = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        stationary_model.weight = streamable_model.weight
        stationary_model.bias = streamable_model.bias

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streamable_model=streamable_model, stationary_model=stationary_model, x=x, slice_size=slice_size)
        print("\tPassed" if is_equal else "\tFailed", "unit test F for Conv1d.")

class ConvTranspose1d():
    """Defines unit tests for the streamable_Conv1d class"""

    def forward_A():
        """Tests the forward method of this class."""

        # Create example data
        time_point_count = 123
        instance_count = 16
        in_channels = 30
        torch.manual_seed(4)
        x = 0*torch.randn(instance_count, in_channels, time_point_count)+1

        # Create streamable model
        out_channels = 21
        kernel_size = 4
        stride = 3
        padding = 2
        dilation = 1
        slice_size = 3
        streamable_model = streamable.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        
        # Create stationary model
        stationary_model = torch.nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        stationary_model.weight = streamable_model.weight
        stationary_model.bias = streamable_model.bias

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streamable_model=streamable_model, stationary_model=stationary_model, x=x, slice_size=slice_size)
        print("\tPassed" if is_equal else "\tFailed", "unit test A for ConvTranspose1d.")

    def forward_B():
        """Tests the forward method of this class."""

        # Create example data
        time_point_count = 123
        instance_count = 19
        in_channels = 26
        x = torch.randn(instance_count, in_channels, time_point_count)

        # Create streamable model
        out_channels = 12
        kernel_size = 7
        stride = 2
        padding = 0
        dilation = 2
        slice_size = 1
        streamable_model = streamable.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation = dilation, padding=padding)
        
        # Create stationary model
        stationary_model = torch.nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        stationary_model.weight = streamable_model.weight
        stationary_model.bias = streamable_model.bias

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streamable_model=streamable_model, stationary_model=stationary_model, x=x, slice_size=slice_size)
        print("\tPassed" if is_equal else "\tFailed", "unit test B for ConvTranspose1d.")

    def forward_C():
        """Tests the forward method of this class."""

        # Create example data
        time_point_count = 7
        instance_count = 19
        in_channels = 26
        x = torch.randn(instance_count, in_channels, time_point_count)

        # Create streamable model
        out_channels = 12
        kernel_size = 6
        stride = 2
        padding = 3
        dilation = 1
        slice_size = 1
        streamable_model = streamable.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation = dilation, padding=padding)
        
        # Create stationary model
        stationary_model = torch.nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        stationary_model.weight = streamable_model.weight
        stationary_model.bias = streamable_model.bias

        # Evaluate (expect an error because the slice size is too small)
        try:
            is_equal = TensorTestUtils.evaluate_prediction(streamable_model=streamable_model, stationary_model=stationary_model, x=x, slice_size=slice_size)
            print("\tFailed unit test C for ConvTranspose1d.")
        except AssertionError as e:
            print("\tPassed unit test C for ConvTranspose1d.")
        except:
            print("\tFailed unit test C for ConvTranspose1d.")

    def forward_D():
        """Tests the forward method of this class."""

        # Create example data
        time_point_count = 123
        instance_count = 19
        in_channels = 26
        x = torch.randn(instance_count, in_channels, time_point_count)

        # Create streamable model
        out_channels = 12
        kernel_size = 7
        stride = 2
        padding = 1
        dilation = 3
        slice_size = 5
        streamable_model = streamable.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        
        # Create stationary model
        stationary_model = torch.nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        stationary_model.weight = streamable_model.weight
        stationary_model.bias = streamable_model.bias

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streamable_model=streamable_model, stationary_model=stationary_model, x=x, slice_size=slice_size)
        print("\tPassed" if is_equal else "\tFailed", "unit test D for ConvTranspose1d.")
    
    def forward_E():
        """Tests the forward method of this class."""

        # Create example data
        time_point_count = 123
        instance_count = 19
        in_channels = 26
        x = torch.randn(instance_count, in_channels, time_point_count)

        # Create streamable model
        out_channels = 12
        kernel_size = 7
        stride = 2
        padding = 10
        dilation = 2
        slice_size = 5
        streamable_model = streamable.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation = dilation, padding=padding)
        
        # Create stationary model
        stationary_model = torch.nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        stationary_model.weight = streamable_model.weight
        stationary_model.bias = streamable_model.bias

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streamable_model=streamable_model, stationary_model=stationary_model, x=x, slice_size=slice_size)
        print("\tPassed" if is_equal else "\tFailed", "unit test E for ConvTranspose1d.")

class Sum():
    def forward_A():
        # Define some variables
        slice_sizes_1 = [0,5,12,13,19,23,30] # Needs to accumulate to same sum as slice_sizes_2
        slice_sizes_2 = [0,2,4,11,15,27,30]
        instance_count = 16
        channel_count = 32
        time_point_count = slice_sizes_1[-1]

        # Create dummy data
        x1 = torch.rand(size=[instance_count, channel_count, time_point_count])
        x2 = torch.rand(size=[instance_count, channel_count, time_point_count])

        # Stream
        streamable_model = streamable.Sum()
        stationary_output = x1 + x2
        streamable_outputs = []
        
        for i in range(len(slice_sizes_2)-1):

            x1_i = x1[:,:,slice_sizes_1[i]:slice_sizes_1[i+1]]
            x2_i = x2[:,:,slice_sizes_2[i]:slice_sizes_2[i+1]]
            is_end_of_stream = i == len(slice_sizes_1) - 2
            if is_end_of_stream: streamable.Module.final_slice()
            streamable_outputs.append(streamable_model(input=[x1_i, x2_i]))
            if is_end_of_stream: streamable.Module.close_stream()

        streamable_output = torch.cat(streamable_outputs, dim=2)

        # Evaluate
        is_equal = np.allclose(a=stationary_output.detach().numpy(), b=streamable_output.detach().numpy(), atol=1e-06)
        print("\tPassed" if is_equal else "\tFailed", "unit test A for Sum")

class Pad1d():
    
    def forward_A():
        # Generate data
        time_point_count = 94
        x = torch.rand([13,15,time_point_count])
        padding = 3
        slice_size = 11
        
        # Generate models
        stationary_model = torch.nn.ReflectionPad1d(padding=padding)
        streamable_model = streamable.Pad1d(padding=padding, mode='reflect')

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streamable_model=streamable_model, stationary_model=stationary_model, x=x, slice_size=slice_size)
        print("\tPassed" if is_equal else "\tFailed", "unit test A for Pad1d.")

    def forward_B():
        # Generate data
        time_point_count = 123
        x = torch.rand([17,11,time_point_count])
        padding = 30
        slice_size = 11
        
        # Generate models
        stationary_model = torch.nn.ReflectionPad1d(padding=padding)
        streamable_model = streamable.Pad1d(padding=padding, mode='reflect')

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streamable_model=streamable_model, stationary_model=stationary_model, x=x, slice_size=slice_size)
        print("\tPassed" if is_equal else "\tFailed", "unit test B for Pad1d.")

    def forward_C():
        # Generate data
        time_point_count = 23
        x = torch.rand([17,11,time_point_count])
        padding = 20
        slice_size = 1
        
        # Generate models
        stationary_model = torch.nn.ReflectionPad1d(padding=padding)
        streamable_model = streamable.Pad1d(padding=padding, mode='reflect')

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streamable_model=streamable_model, stationary_model=stationary_model, x=x, slice_size=slice_size)
        print("\tPassed" if is_equal else "\tFailed", "unit test C for Pad1d.")

    def forward_D():
        # Generate data
        time_point_count = 23
        x = torch.rand([17,11,time_point_count])
        padding = [20,0]
        slice_size = 1
        
        # Generate models
        stationary_model = torch.nn.ReflectionPad1d(padding=padding)
        streamable_model = streamable.Pad1d(padding=padding, mode='reflect')

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streamable_model=streamable_model, stationary_model=stationary_model, x=x, slice_size=slice_size)
        print("\tPassed" if is_equal else "\tFailed", "unit test D for Pad1d.")

    def forward_E():
        # Generate data
        time_point_count = 23
        x = torch.rand([17,11,time_point_count])
        padding = [0,11]
        slice_size = 1
        
        # Generate models
        stationary_model = torch.nn.ReflectionPad1d(padding=padding)
        streamable_model = streamable.Pad1d(padding=padding, mode='reflect')

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streamable_model=streamable_model, stationary_model=stationary_model, x=x, slice_size=slice_size)
        print("\tPassed" if is_equal else "\tFailed", "unit test E for Pad1d.")

    def forward_F():
        # Generate data
        time_point_count = 23
        x = torch.rand([17,11,time_point_count])
        padding = [5,8]
        slice_size = 2
        
        # Generate models
        stationary_model = torch.nn.ReflectionPad1d(padding=padding)
        streamable_model = streamable.Pad1d(padding=padding, mode='reflect')

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streamable_model=streamable_model, stationary_model=stationary_model, x=x, slice_size=slice_size)
        print("\tPassed" if is_equal else "\tFailed", "unit test F for Pad1d.")

    def forward_G():
        # Generate data
        time_point_count = 23
        x = torch.rand([17,11,time_point_count])
        padding = [5,8]
        slice_size = 7
        
        # Generate models
        stationary_model = torch.nn.ReflectionPad1d(padding=padding)
        streamable_model = streamable.Pad1d(padding=padding, mode='reflect')

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streamable_model=streamable_model, stationary_model=stationary_model, x=x, slice_size=slice_size)
        print("\tPassed" if is_equal else "\tFailed", "unit test G for Pad1d.")

    def forward_H():
        # Generate data
        time_point_count = 23
        x = torch.rand([17,11,time_point_count])
        padding = [5,8]
        slice_size = 7
        
        # Generate models
        stationary_model = torch.nn.ConstantPad1d(padding=padding, value=0)
        streamable_model = streamable.Pad1d(padding=padding, mode='constant')

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streamable_model=streamable_model, stationary_model=stationary_model, x=x, slice_size=slice_size)
        print("\tPassed" if is_equal else "\tFailed", "unit test H for Pad1d.")

    def forward_I():
        # Generate data
        time_point_count = 23
        x = torch.rand([17,11,time_point_count])
        padding = [5,8]
        slice_size = 7
        
        # Generate models
        stationary_model = torch.nn.ReplicationPad1d(padding=padding)
        streamable_model = streamable.Pad1d(padding=padding, mode='replicate')

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streamable_model=streamable_model, stationary_model=stationary_model, x=x, slice_size=slice_size)
        print("\tPassed" if is_equal else "\tFailed", "unit test I for Pad1d.")


if __name__ == "__main__":
    print("\nUnit tests for models.streamable_modules.")
    
    # Conv1d
    Conv1d.forward_A()
    Conv1d.forward_B()
    Conv1d.forward_C()
    Conv1d.forward_D()
    Conv1d.forward_E()
    Conv1d.forward_F()
    
    # ConvTranspose1d
    ConvTranspose1d.forward_A()
    ConvTranspose1d.forward_B()
    ConvTranspose1d.forward_C()
    ConvTranspose1d.forward_D()
    ConvTranspose1d.forward_E()

    # Sum
    Sum.forward_A()
    
    # Pad1d
    Pad1d.forward_A()
    Pad1d.forward_B()
    Pad1d.forward_C()
    Pad1d.forward_D()
    Pad1d.forward_E()
    Pad1d.forward_F()
    Pad1d.forward_G()
    Pad1d.forward_H()
    Pad1d.forward_I()
    