import sys
sys.path.append(".")
from models import neural_networks as mnn
from models import fitter as mft
import torch

class Fitter():
    """This class provides unit tests for fitter.Fitter."""

    @staticmethod
    def test_A():
        # Shapes 
        instance_count = 32
        time_frame_count = 64
        input_feature_count = 3
        output_feature_count = 5
        
        # Create data
        x = torch.ones(size=(instance_count,time_frame_count,input_feature_count))
        y = torch.zeros(size=(instance_count,time_frame_count,output_feature_count))

        # Neural network
        stationary_neural_network = mnn.Dense(input_feature_count=input_feature_count, output_feature_count=output_feature_count, is_streamable=False)

        # Fit
        optimizer = torch.optim.Adam(params=stationary_neural_network.parameters(), lr=0.01)
        fitter = mft.Fitter(is_streamable=False)
        train_losses, validation_losses = fitter.fit(stationary_neural_network=stationary_neural_network, x=x, y=y, 
            loss_function=torch.nn.functional.mse_loss, optimizer=optimizer, epoch_count=100, shuffle_seed=42)

        # Predict
        y_hat = stationary_neural_network.predict(x=x)

        # Evaluate
        is_equal = y.size() == y_hat.size()
        if is_equal:
            is_equal = torch.allclose(input=y, other=y_hat, atol=0.1)

        # Log
        print("\tPassed" if is_equal else "\tFailed", f"unit test A for Fitter.")

    @staticmethod
    def test_B():
        # Shapes 
        slice_count = 8
        slice_size = 16
        time_frame_count = slice_count * slice_size
        input_feature_count = 3
        output_feature_count = 5
        
        # Create data
        x = torch.ones(size=(time_frame_count,input_feature_count))
        y = torch.zeros(size=(time_frame_count,output_feature_count))

        # Neural network
        stationary_neural_network = mnn.Dense(input_feature_count=input_feature_count, output_feature_count=output_feature_count, is_streamable=False)

        # Fit stream
        optimizer = torch.optim.Adam(params=stationary_neural_network.parameters(), lr=0.01)
        fitter = mft.Fitter(is_streamable=True)
        
        for i in range(slice_count):
            # Slice data
            x_i = x[i*slice_size:(i+1)*slice_size,:]
            y_i = y[i*slice_size:(i+1)*slice_size,:]
        
            # Fit on slice
            train_losses, validation_losses = fitter.fit(stationary_neural_network=stationary_neural_network, x=x_i, y=y_i, 
                loss_function=torch.nn.functional.mse_loss, optimizer=optimizer, epoch_count=10, shuffle_seed=42, is_final_slice=i==slice_count-1)

        # Predict
        y_hat = stationary_neural_network.predict(x=x)

        # Evaluate
        is_equal = y.size() == y_hat.size()
        if is_equal:
            is_equal = torch.allclose(input=y, other=y_hat, atol=0.1)

        # Log
        print("\tPassed" if is_equal else "\tFailed", f"unit test B for Fitter.")

if __name__ == "__main__":
    print("\nUnit tests for models.fitter.")

    Fitter.test_A()
    Fitter.test_B()