from timeflux.core.node import Node
import numpy as np, pandas as pd
import sys, torch, os
sys.path.append('.')
from models import neural_networks

class Predictor(Node):
    """This node predicts the speech spectrogram based on eeg.
    It expects as input streams of the aligned eeg feature time course, the label time course and the fitter loss. 
    It has one output stream for the speech spectrogram."""

    def __init__(self, name: str, eeg_stream_name: str, speech_stream_name: str, label_stream_name: str, neural_network_type: str, parameters_path: str) -> object:
        """Constructor for this class.
        
        Inputs:
        - name: The name assigned to this node.
        - eeg_stream_name, speech_stream_name, label_stream_name: The names of the streams. Each such name is used to identify a stream from the input ports.
        - neural_network_type: The type of the neural network to be trained, e.g. Dense or Convolutional. This type will be taken from models.neural_networks.
        - parameters_path: The path to the file where the neural network parameters shall be stored after each fit operation. This path is relative to the parameters directory.
       
        Outputs:
        - self: The initialized instance."""
        
        # Super
        super(Predictor, self).__init__()

        # Copy attributes
        self.name = name
        self.__eeg_stream_name__ = eeg_stream_name
        self.__speech_stream_name__ = speech_stream_name
        self.__label_stream_name__ = label_stream_name
        self.__neural_network_type__ = neural_network_type
        self.parameters_path = parameters_path

        # Set variable for neural network
        self.streamable_neural_network = None

        # Set up buffer
        self.__buffer__ = {eeg_stream_name: None, label_stream_name: None, speech_stream_name: None, 'losses': None}

        # Set loss change 
        self.__previous_loss__ = None

    def update(self):
        # Clear buffer
        for key in self.__buffer__.keys(): self.__buffer__[key] = None

        # Iterate ports
        for _, _, port in self.iterate("i*"):
            if (port.ready()):
                # Input validity
                assert port.meta.get('stream_name') in self.__buffer__.keys(), f"Expected stream_name meta variable of input to be in {self.__buffer__.keys()}, but received{port.meta.get('stream_name')}"

                # Extract inputs
                self.__buffer__[port.meta.get('stream_name')] = port.data

        # Exit if data incomplete
        valid = True
        for key in self.__buffer__.keys(): valid = valid and not self.__buffer__[key] is None
        if not valid: return super().update()

        # Extract data from buffer
        eeg = torch.Tensor(self.__buffer__[self.__eeg_stream_name__].values)
        speech = self.__buffer__[self.__speech_stream_name__] # We only convert speech to torch if the neural network is not used
        labels = self.__buffer__[self.__label_stream_name__]

        # Check whether the loss changed between previous and current slice
        loss_changed = self.__did_loss_change__(losses=self.__buffer__['losses']['validation'])
        
        # Ensure neural network exists
        if self.streamable_neural_network == None:
            # Shapes
            eeg_feature_count = eeg.size()[-1]; speech_feature_count = speech.shape[-1]
            
            # Get the actual type of the neural network
            neural_network_type = getattr(neural_networks, self.__neural_network_type__)
            
            # Initialize
            self.streamable_neural_network = neural_network_type(input_feature_count=eeg_feature_count, output_feature_count=speech_feature_count, is_streamable=True)
            
        # Load the latest model
        if loss_changed:
            self.streamable_neural_network.load_state_dict(torch.load(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'parameters', self.parameters_path + '.pt'))))
            print("Loaded")
        
        # Scale
        eeg=(eeg-torch.mean(eeg,axis=1).unsqueeze(1))/torch.std(eeg,axis=1).unsqueeze(1)

        # Predict
        speech_predicted = self.streamable_neural_network.predict(x=eeg.unsqueeze(0), is_final_slice=False).squeeze()

        # Output
        self.o.data = pd.DataFrame(speech_predicted.detach().numpy())

    def __did_loss_change__(self, losses: pd.Series) -> bool:
        """Mutating method that determines whether the loss changed between the last time frame of the previous loss slice and any of the current time frames.
        
        Precondition:
        - self.__previous_loss__ may be None or a float.

        Inputs:
        - losses: the current series of losses. Maybe be empty.
        
        Postcondition:
        - self.__previous_loss__ is be None (if this loss series was empty) or equal to the last entry of losses (if this losses contained an entry)."""
        
        # Extract values
        losses = losses.values

        loss_changed = False
        # If losses contains rows
        if len(losses):
            # Check if previously seen loss is different from the first new loss
            if self.__previous_loss__ != None: loss_changed = self.__previous_loss__ != losses[0]

            # Check if there is a change within the new losses
            zeros = np.zeros(len(losses)-1)
            loss_changed = loss_changed or not np.allclose(zeros, losses[:-1] - losses[1:])

            # Update previous loss
            self.__previous_loss__ = losses[-1]

        # Outputs
        return loss_changed
