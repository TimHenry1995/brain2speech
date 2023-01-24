import sys
sys.path.append(".")
from models import neural_networks as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

class Dense():
    """This class provides unit tests for neural_networks.Dense."""

    @staticmethod
    def test_A():
        # Shapes
        instance_count = 16
        time_frame_count = 8
        input_feature_count = 4
        output_feature_count = 2
        
        # Create neural network
        dense = nn.Dense(input_feature_count=input_feature_count, output_feature_count=output_feature_count, is_streamable=False)

        # Create data
        x = torch.ones(size=(instance_count,time_frame_count,input_feature_count))
        y = torch.zeros(size=(instance_count,time_frame_count,output_feature_count))

        # Fit
        optimizer = torch.optim.Adam(params=dense.graph.parameters(), lr=0.1)
        train_losses, validation_losses = dense.fit(x=x, y=y, loss_function=torch.nn.functional.mse_loss, optimizer=optimizer, 
            epoch_count=40, shuffle_seed=42)

        # Predict
        y_hat = dense.predict(x=x)

        # Evaluate
        is_equal = torch.allclose(input=y, other=y_hat, atol=0.1)

        # Log
        print("\tPassed" if is_equal else "\tFailed", "unit test A for Dense.")

    
    @staticmethod
    def test_B():
        # Shapes
        instance_count = 64
        time_frame_count = 8
        input_feature_count = 4
        output_feature_count = 2
        
        # Create neural network
        dense = nn.Dense(input_feature_count=input_feature_count, output_feature_count=output_feature_count, is_streamable=True)

        # Create data
        x = torch.ones(size=(instance_count,time_frame_count,input_feature_count))
        y = torch.zeros(size=(instance_count,time_frame_count,output_feature_count))

        # Fit stream
        optimizer = torch.optim.Adam(params=dense.graph.parameters(), lr=0.1)
        step_size = 16
        step_count = instance_count // step_size
        for i in range(step_count):
            x_i, y_i = x[i*step_size:(i+1)*step_size,:,:], y[i*step_size:(i+1)*step_size,:,:] 
            train_losses, validation_losses = dense.fit(x=x_i, y=y_i, loss_function=torch.nn.functional.mse_loss, optimizer=optimizer, 
                epoch_count=10, shuffle_seed=42, is_final_slice=i==step_count-1)

        # Predict
        y_hat = dense.predict(x=x)

        # Evaluate
        is_equal = torch.allclose(input=y, other=y_hat, atol=0.1)

        # Log
        print("\tPassed" if is_equal else "\tFailed", "unit test B for Dense.")

if __name__ == "__main__":
    print("\nUnit tests for models.neural_networks.")

    # Dense
    #Dense.test_A()
    Dense.test_B()
