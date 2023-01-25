from timeflux.core.node import Node
import matplotlib.pyplot as plt, numpy as np
from typing import List
import torch

class Fitter(Node):
    def __init__(self) -> object:
        # Construct model
        pass

    def update(self) -> None:
        # Fit
        pass

    def terminate(self) -> None:
        return super().terminate()

    def plot_x_target_and_output(self, x: torch.Tensor, target: torch.Tensor, output: torch.Tensor, labels: List[str], pause_string: str, path: str) -> None:
        """Plots x, the target and output.
        Inputs:
        - x: Input EEG time series. Shape == [time frame count, eeg channel count].
        - target: Desired spectrogram. Shape == [time frame count, mel channel count].
        - output: Obtained spectrogram. Shape == [time frame count, mel channel count].
        - labels: The labels that indicate for each time frame of x, target and output which word was present at that time. Length == time frame count.
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
