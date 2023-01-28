from timeflux.core.node import Node
import sys, torch, pandas as pd
sys.path.append('.')
import models.neural_networks as mnn

class VocGan(Node):
    """This node transform from a mel speech spectrogram to a waveform. It expects a single stream
    of mel spectrogram temporal slices with 80 features in the form of a pandas DataFrame. 
    It has one output stream for the waveform in the form of a pandas DataFrame."""
    
    def __init__(self, name):
        super(VocGan, self).__init__()

        # Copy attributes
        self.name = name

        # Create neural_netwok
        self.streamable_neural_netwok = mnn.VocGan.load(is_streamable=True)


    def update(self):
        # Early exit
        if not self.i.ready(): return super().update()
        
        # Extract the current slice
        spectrogram = torch.Tensor(self.i.data.values).permute(dims=[1,0])
        waveform, processing_time = self.streamable_neural_netwok.mel_spectrogram_to_waveform(mel_spectrogram=spectrogram, is_final_slice=False)
            
        # Output
        self.o.data = pd.DataFrame(waveform)