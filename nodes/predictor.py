from timeflux.core.node import Node


class Predictor(Node):
    """This node predicts the speech spectrogram based on eeg.
    It expects as input streams the eeg feature time course and the label time course. 
    It has one output stream for the speech spectrogram."""

    def __init__(self, name: str, eeg_stream_name: str, label_stream_name: str, neural_network_type: str, parameters_path: str) -> object:
        """Constructor for this class.
        
        Inputs:
        - name: The name assigned to this node.
        - eeg_stream_name, label_stream_name: The names of the streams. Each such name is used to identify a stream from the input ports.
        - neural_network_type: The type of the neural network to be trained, e.g. Dense or Convolutional. This type will be taken from models.neural_networks.
        - parameters_path: The path to the file where the neural network parameters shall be stored after each fit operation. This path is relative to the parameters directory.
       
        Outputs:
        - self: The initialized instance."""
        
        # Super
        super(Predictor, self).__init__()

        # Copy attributes
        self.name = name
        self.eeg_stream_name = eeg_stream_name
        self.