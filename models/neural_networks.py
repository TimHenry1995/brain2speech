import torch, scipy
from scipy.io import wavfile
from abc import ABC
from sklearn.utils import shuffle
from typing import Union, List, Dict, Tuple
import sys, os, time, math
sys.path.append(".")
from models import streamable_modules as streamable
from models import stationary_modules as stationary
from models import module_converter as mc
from models import utilities as mut
from plugins import hparams, stft
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import copy

class NeuralNetwork(torch.nn.Module, ABC):
    """This class provides functionality for neural networks that can be used in streamable and stationary mode."""

    def __init__(self, is_streamable: bool, name: str = "Neural Network") -> object:
        """Constructor for this class. The inheriting subclasses are expected to behave as follows:
        0. Call this super initializer.
        1. Create a dictionary that gives for a module name its corresponding stationary module, e.g. convertible_modules = {'linear_1': torch.nn.Linear(...), 'linear_2': torch.nn.Linear(...)}
        2. Convert them to streamable modules if needed using: if is_streamable: convertible_modules = NeuralNetwork.__to_streamable__(modules=convertible_modules)
        3. Save the convertible modules such that they can be used for your later computations. 
            E.g. if you use a sequential module, provide them to the sequential module or 
            if you use the modules as attributes of your neural network, e.g. query, key, value 
            layers in an attention mechanism then save these modules one by one as attributes of self.
            Do not simply save the convertible_modules dictionary as an attribute because then torch is 
            not guaranteed to find them for saving, loading or simply getting the parameters of you neural network.
        
        Inputs:
        - is_streamable: Indicates whether this neural network shall be used in streamable or stationary mode.
        - name: The name of this neural network.
        """
        super(NeuralNetwork, self).__init__()

        self.__is_streamable__ = is_streamable
        self.name = name

    @staticmethod
    def __to_streamable__(modules: Dict[str,torch.nn.Module]) -> Dict[str,streamable.Module]:
        """Converts modules from stationary to streamable. Assumes that the module_conveter has a mapping for that module.
        
        Inputs:
        - modules: provides for a module name key a stationary module, e.g. Linear, Conv1d, Sum. 
        
        Outputs:
        - modules: the same dictionary with its modules replaced by their streamable equivalents."""

        # Replace each module with its alternative
        for name, module in modules.items():
            modules[name] = mc.ModuleConverter.stationary_to_streamable(module=module)

        # Outputs
        return modules

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
        y_hat = self(x)

        # Handle streamable mode
        if self.__is_streamable__ and is_final_slice: streamable.Module.close_stream()

        # Outputs
        return y_hat

class Dense(NeuralNetwork):
    """This class is a dense neural network."""

    def __init__(self, input_feature_count: int, output_feature_count: int, is_streamable: bool = False, name: str = "Dense") -> object:
        """Constructor for this class.
        
        Inputs:
        - input_feature_count: number of input features.
        - output_feature_count: number of output features.
        - is_streamable: Indicates whether this neural network shall be used in streamable or stationary mode.
        - name: the name of this neural network.
        """
    
        # Following the super class instructions for initialization
        # 0. Super
        super(Dense, self).__init__(is_streamable=is_streamable, name=name)

        # 1. Create a dictionary of stationary modules
        convertible_modules = { 'linear_1': torch.nn.Linear(input_feature_count, output_feature_count),
                                'linear_2': torch.nn.Linear(output_feature_count, output_feature_count)}
        
        # 2. Convert them to streamable modules
        if is_streamable: convertible_modules = NeuralNetwork.__to_streamable__(modules=convertible_modules)
        
        # 3. Save the convertible modules for later computations     
        self.sequential = torch.nn.Sequential(convertible_modules['linear_1'], torch.nn.ReLU(), convertible_modules['linear_2'])
  
    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Predict
        y_hat = self.sequential(x)

        # Outputs
        return y_hat
        
class Convolutional(NeuralNetwork):
    """This class is a convolutional neural network."""

    def __init__(self, input_feature_count: int, output_feature_count: int, is_streamable: bool = False, name: str = "Convolutional") -> object:
        """Constructor for this class.
        
        Inputs:
        - input_feature_count: number of input features.
        - output_feature_count: number of output features.
        - is_streamable: indicates whether this neural network shall be used in streamable or stationary mode.
        - name: the name of this neural network
        """
    
        # Following the super class instructions for initialization
        # 0. Super
        super(Convolutional, self).__init__(is_streamable=is_streamable, name=name)

        # 1. Create a dictionary of stationary modules
        convertible_modules = {
            'convolutional_1': torch.nn.Conv1d(input_feature_count, input_feature_count, kernel_size=8, dilation=8),
            'convolutional_2': torch.nn.Conv1d(input_feature_count, input_feature_count, kernel_size=4, dilation=4),
            'convolutional_3': torch.nn.Conv1d(input_feature_count, output_feature_count, kernel_size=2, dilation=2),
            'pad_1': torch.nn.ConstantPad1d(padding=[(8-1)*8,0], value=0),
            'pad_2': torch.nn.ConstantPad1d(padding=[(4-1)*4,0], value=0),
            'pad_3': torch.nn.ConstantPad1d(padding=[(2-1)*2,0], value=0)}
        
        # 2. Convert them to streamable modules
        if is_streamable: convertible_modules = NeuralNetwork.__to_streamable__(modules=convertible_modules)
        
        # 3. Save the convertible modules for later computations 
        self.sequential = torch.nn.Sequential(
            Transpose(),
            convertible_modules['pad_1'],
            convertible_modules['convolutional_1'], torch.nn.ReLU(),
            convertible_modules['pad_2'],
            convertible_modules['convolutional_2'], torch.nn.ReLU(),
            convertible_modules['pad_3'],
            convertible_modules['convolutional_3'],
            Transpose()
        )
  
    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Predict
        y_hat = self.sequential(x)

        # Outputs
        return y_hat

class Transpose(torch.nn.Module):
    """Transposes inputs with along the final two dimensions"""
    
    def forward(self, x):

        return x.permute(dims=list(range(len(x.size())-2)) + [-1,-2])

class MemoryDense(NeuralNetwork):
    """This class is a dense neural network that is suited for time series.
    It keeps a memory of past time frames and for each prediction uses dense connections from the entire memory to the output."""

    def __init__(self, input_feature_count: int, output_feature_count: int, step_size: int, step_count: int, layer_count: int, is_streamable: bool = False, name = "MemoryDense") -> object:
        """Constructor for this class.
        
        Inputs:
        - input_feature_count: number of input features.
        - output_feature_count: number of output features.
        - step_size: the number of steps until the previous time frame is considered. E.g. if set to 1 then the previous time frame is considered. If set to 2 then one time frame will be skipped.
        - step_count: the number of previous time frames with step size step_size that should influence the predition. If set to 1 only the current time frame is used. If set to 2 then also the previous time frame is used.
        - layer_count: integer at least 1. The number of layers.
        - is_streamable: indicates whether this neural network shall be used in streamable or stationary mode.
        - name: the name of this neural network.
        """
    
        # Following the super class instructions for initialization
        # 0. Super
        super(MemoryDense, self).__init__(is_streamable=is_streamable, name=name)

        # 1. Create a dictionary of stationary modules. The memory effect is created by a convolution 
        convertible_modules = { 
            'pad_1': torch.nn.ConstantPad1d(padding=[(step_count-1)*step_size,0], value=0),
            'conv_1': torch.nn.Conv1d(in_channels=input_feature_count, out_channels=output_feature_count, kernel_size=step_count, dilation=step_size),
            }
        for i in range(1,layer_count):
            convertible_modules[f"linear_{i+1}"] = torch.nn.Linear(output_feature_count, output_feature_count)

        # 2. Convert them to streamable modules
        if is_streamable: convertible_modules = NeuralNetwork.__to_streamable__(modules=convertible_modules)
        
        # 3. Save the convertible modules for later computations     
        extra_layers = []
        for i in range(1,layer_count):
            extra_layers.append(torch.nn.ReLU())
            extra_layers.append(convertible_modules[f"linear_{i+1}"])
        self.sequential = torch.nn.Sequential(Transpose(), convertible_modules['pad_1'], convertible_modules['conv_1'], Transpose(), *extra_layers)
  
    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Predict
        y_hat = self.sequential(x)

        # Outputs
        return y_hat

class Recurrent(NeuralNetwork):
    """This class implements a recurrent neural network"""

    def __init__(self, input_feature_count: int, output_feature_count: int, is_streamable: bool = False, name: str = "Recurrent") -> object:
        """Constructor for this class.
        
        Inputs:
        - input_feature_count: number of input features.
        - output_feature_count: number of output features.
        - is_streamable: Indicates whether this neural network shall be used in streamable or stationary mode.
        - name: the name of this neural network.
        """
    
        # Following the super class instructions for initialization
        # 0. Super
        super(Recurrent, self).__init__(is_streamable=is_streamable, name=name)

        # 1. Create a dictionary of stationary modules
        convertible_modules = { 'gru': stationary.GRU(input_size=input_feature_count, hidden_size=input_feature_count, num_layers=1, batch_first=True),
                                'linear': torch.nn.Linear(input_feature_count, output_feature_count)}
        
        # 2. Convert them to streamable modules
        if is_streamable: convertible_modules = NeuralNetwork.__to_streamable__(modules=convertible_modules)
        
        # 3. Save the convertible modules for later computations     
        self.gru = convertible_modules['gru']
        self.linear = convertible_modules['linear']

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Predict
        y_hat = self.gru(x)
        y_hat = self.linear(y_hat)

        # Outputs
        return y_hat

class Attention(NeuralNetwork):
    """This class provides an attention module that uses a query (e.g. EEG time course) to select from a value (e.g. speech spectrogram) 
    using a similarity of intermediate representations for query and a key (e.g. EEG time course). Here, x_key and x_value may relate to one another with
    a temporal offset. That is, timepoint r in x_key might relate mostly to time point r + delta in x_value. Assuming that delta is not known multiple deltas will be 
    tried and the network has to choose which one is optimal."""

    def __init__(self, query_feature_count: int, hidden_feature_count: int, x_key: torch.Tensor, x_value: torch.Tensor, labels: List[str], pause_string: str, step_count: int = 1, step_size: int = 1, is_streamable: bool = False, name: str = "Attention") -> object:
        """Constructor for this class.
        
        Inputs:
        - query_feature_count: The number of features of the query, i.e. the time series for which similiar time frames are searched.
        - hidden_feature_count: The number of hidden features, i.e. the representation of query and hidden used to estimate their similarity.
        - x_key: The time course (e.g. EEG) whose similarity with the query time course decides which value vector to choose from x_value. 
            Shape == [time frame count, channel count].
        - x_value: The time course (e.g. speech spectrogram) from which value vectors are chosen based on the simlarity of x_query and x_key. Indexing assumed to be in sync with that of x_key.
            Shape == [time frame count, channel count].
            The time frame count has to be the same as for x_key but the channel count may be different.
        - labels: The labels that indicate for each time frame which word was shown on screen.
        - pause_string: The string used to indicate pauses between words.
        - step_count : Integer at least 1. The number of shifts between x_key and x_value that should be tried. 
            For a shift count of 1 each time frame of x_key will directly relate to its corresponing time frame in x_key.
            For a shift count of 2 there will be an operation where the attention matrix is shifted step_size many units into the future 
            (along both the query and the value axes). Hence, if time frame s of the query attends time frame r of the value then time frame 
            r + step_size is assigned from the value to time frame s + step_size of the output. If shift count is set to 3 then 
            there will be yet another attention operation where the step_size is doubled etc..
        - step_size: Integer at least 1. The number of steps by which timeframes of the attention matrix are shifted to the future along
            both the query and the value axes to account for temporal offsets between x_key and x_value. 
        - is_streamable: Indicates whether this neural network shall be used in streaming mode.
        - name: The name of the neural network.

        Outputs:
        - The instance of this class."""

        # Input validity
        assert type(x_key) == torch.Tensor, f"Expected x_key to have type torch.Tensor, not {type(x_key)}"
        assert len(x_key.size()) == 2, f"Expected x_key to have shape [time frame count, channel count], received {x_key.size()}"
        assert type(x_value) == torch.Tensor, f"Expected x_value to have type torch.Tensor, not {type(x_value)}"
        assert len(x_value.size()) == 2, f"Expected x_value to have shape [time frame count, channel count], received {x_value.size()}"
        assert type(labels) == type([""]), f'Expected labels to have type {type([""])} but received {type([""])}'
        assert x_key.size()[0] == x_value.size()[0] and x_value.size()[0] == len(labels), f"Exepected x_key, x_value and labels to have same number of time frame along axis 0, received {x_key.size()}, {x_value.size()}, {len(labels)} respectively."
        assert step_count >= 1, f"The step_count is expected to be at least 1. Received {step_count}"
        assert step_size >= 1, f"The step_size is expected to be at least 1. Received {step_size}"

        # Fields
        self.step_count = step_count
        self.step_size = step_size

        # Save key and value
        self.x_key = copy.deepcopy(x_key) 
        self.x_value = copy.deepcopy(x_value) 

        # Position encode self.x_key 
        self.x_key = mut.reshape_by_label(x=x_key, labels=labels, pause_string=pause_string) 
        key_instances_per_batch, key_time_frames_per_instance, key_feature_count = self.x_key.size()
        P = Attention.__get_position_encoding__(instances_per_batch=key_instances_per_batch, time_frames_per_instance=key_time_frames_per_instance, feature_count=key_feature_count)
        self.x_key = mut.undo_reshape_by_label(y=self.x_key + P, labels=labels, pause_string=pause_string)

        # Following the super class instructions for initialization
        # 0. Super
        super(Attention, self).__init__(is_streamable=is_streamable, name=name)

        # 1. Create a dictionary of stationary modules
        convertible_modules = { 'query_key_layer': torch.nn.Linear(in_features=query_feature_count, out_features=hidden_feature_count, bias=False),
                                'shift_layer': torch.nn.Linear(in_features=step_count, out_features=1, bias=False)}
        
        # 2. Convert them to streamable modules
        if is_streamable: convertible_modules = NeuralNetwork.__to_streamable__(modules=convertible_modules)
        
        # 3. Save the convertible modules for later computations     
        self.query_key_layer = convertible_modules['query_key_layer']
        self.query_key_layer.weight = torch.nn.parameter.Parameter(torch.eye(n=hidden_feature_count, m=query_feature_count))
        self.shift_layer = convertible_modules['shift_layer']
        self.shift_layer.weight = torch.nn.parameter.Parameter(torch.ones(size=(1, step_count))/step_count)

    @staticmethod
    def __stack_attention__(attention: torch.Tensor, step_count: int = 1, step_size: int = 1) -> torch.Tensor:
        """Shifts the time frames of the attention matrix step_count-1 many times by step_size many time frames into
        the future along both the query and the value axes. Zero padding is applied to the past time points to ensure the 
        number of time frames stay constant. The resulting matrices are stacked.
        
        Inputs:
        - attention: The matrix to be shifted. Shape == [instances per batch, time frame count query, time frame count value].
        - step_count: Integer > 1. The number of shifts to be performed. If set to 1 then just the original x_value will be returned.
        - step_size: Integer > 1. The number of time frames between two adjacent shifts - 1.
        
        Outputs:
        - attention_stack: The stack of shifted attention matrices. Shape == [instances per batch, step_count, time frame count query, time frame count value]
        """

        # Shapes
        instances_per_batch, time_frame_count_query, time_frame_count_value = attention.size()

        # Pad it along both axes
        k = (step_count - 1)*step_size
        attention_padded = torch.zeros(size=[instances_per_batch, time_frame_count_query + k, time_frame_count_value + k])
        attention_padded[:, k:, k:] = attention
        
        # Shift and stack
        xs = [None] * (step_count)
        for p in range(len(xs)): 
            a = p*(step_size)
            b = a + time_frame_count_query
            c = a + time_frame_count_value
            xs[p] = attention_padded[:,a:b,a:c].unsqueeze(1)
        attention_stack = torch.cat(xs, dim=1) # Instance axis

        # Outputs
        return attention_stack

    @staticmethod
    def __get_position_encoding__(instances_per_batch: int, time_frames_per_instance: int, feature_count: int, n: int = 10000) -> torch.Tensor:
        """Generates a position encoding, i.e. a matrix of damped sinusoids that encode time across rows.
        
        Inputs:
        - instances_per_batch: The number of instances that should be in a batch.
        - time_frames_per_instance: The number of time frames for the output matrix.
        - feature_count: The number of features along which a time vector shall expand.
        
        Outputs:
        - x: The positional encoding of shape [instances_per_batch, time_frames_per_instance, feature_count]"""

        # Iterate time frames
        x_value = np.zeros((time_frames_per_instance, feature_count))
        for k in range(time_frames_per_instance):
        
            # Iterate features
            for i in np.arange(int(feature_count/2)):
            
                # Generate damped sinusoids
                denominator = np.power(n, 2*i/feature_count)
                x_value[k, 2*i] = np.sin(k/denominator)
                x_value[k, 2*i+1] = np.cos(k/denominator)
      
        # Typing
        x_value = torch.Tensor(np.array(x_value, dtype=np.float32))

        # Repeat for each instance
        x_value = x_value.repeat(repeats=(instances_per_batch, 1, 1))

        # Outputs
        return x_value
    
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Inputs:
        - x: The query of shape [instances per batch, time frames per query, query feature count].
            
        Outputs:
        - y_hat: The result of querying self.x_value with x_query via an attention matrix obtained from self.x_query and self.x_key. 
            Shape == [instance count per batch, time frames per query, self.x_value feature count]
        """
        # Input validity
        assert type(x) == torch.Tensor, f"Expected x to have type torch.Tensor, not {type(x)}"
        assert len(x.size()) == 3 or len(x.size()) == 2, f"Expected x to have shape [instances per batch, time frames per query, query feature count] or [time frames per query, query feature count]], received {x.size()}"

        # Rename input
        x_query = x

        # Ensure shape
        if len(x_query.size()) == 2: x_query = x_query.unsqueeze(0) # shape == [instances per batch, time frames per query, query feature count]
        
        # Counts
        query_instances_per_batch, query_time_frames_per_instance, query_feature_count = x_query.size()
        
        # Repeat x_key
        x_key = self.x_key.repeat(repeats=(query_instances_per_batch, 1, 1)) # shape == [instances per batch, time frames per key, key feature count]
        
        # Position encoding with same shape as query
        p_query = Attention.__get_position_encoding__(instances_per_batch=query_instances_per_batch, time_frames_per_instance=query_time_frames_per_instance, feature_count=query_feature_count)
      
        # Transform
        Q = F.relu(self.query_key_layer(x_query + p_query)) # Shape == [instances per batch, time frames per query, self.hidden_node_count]
        K = F.relu(self.query_key_layer(x_key)) # Shape == [instances per batch, time frames per key, self.hidden_node_count]
        
        # Form attention matrix
        A = Q.matmul(K.permute(0,2,1))
        A = A / (torch.std(A, dim=-1).unsqueeze(-1)+ 1e-5) # The epsilon avoids division by zero
        A = F.softmax(A, dim=-1) # Shape == [instances per batch, time frames per query, time frames per key]
        A = Attention.__stack_attention__(attention=A, step_count=self.step_count, step_size=self.step_size) # Shape == [instances per batch, shift count, time frames per query, time frames per key]
        
        # Select from x_value by attention (remember that x_value and x_key have the same time frame count)
        x_value = self.x_value.unsqueeze(0).unsqueeze(1).repeat(repeats=(query_instances_per_batch, self.step_count, 1, 1)) # Shape == [instances per batch, shift count, time frames per value, value feature count]
        y_hat = A.matmul(x_value) # Shape == [instances per batch, shift count, time frames per query, value feature count]

        # Collapse along shift axis
        y_hat = y_hat.permute(dims=(0,2,3,1)) # Now shift axis is last 
        y_hat = self.shift_layer(y_hat).squeeze() # Shape == [instances per batch, time frames per query, value feature count]
        
        # Outputs
        return y_hat

class ResStack(NeuralNetwork):
    """This class implements a residual stack."""

    def __init__(self, is_streamable: bool, feature_count: int, dilation: int =1, name: str = "ResStack"):
        """Constructor for this class.
        
        Inputs:
        - is_streamable: indicates whether this neural network shall be used in streamable or stationary mode.
        - feature_count: the number of features per time frame for input and output.
        - dilation: the dilation for convolution operations.
        - name: the name of this neural network.
        """
        
        # 0. Super
        super(ResStack, self).__init__(is_streamable=is_streamable, name=name)

        # 1. Create a dictionary of stationary modules
        convertible_modules = {
            'reflection_pad_1': torch.nn.ReflectionPad1d(padding=dilation),
            'conv1d_1': torch.nn.Conv1d(in_channels=feature_count, out_channels=feature_count, kernel_size=3, dilation=dilation),
            'conv1d_2': torch.nn.Conv1d(in_channels=feature_count, out_channels=feature_count, kernel_size=1),
            'conv1d_3': torch.nn.Conv1d(in_channels=feature_count, out_channels=feature_count, kernel_size=1),
            'sum': stationary.Sum()
            }
        
        # 2. Convert them to streamable modules
        if is_streamable: convertible_modules = NeuralNetwork.__to_streamable__(modules=convertible_modules)
        
        # 3. Save the convertible modules for later computations         
        self.block = torch.nn.Sequential(
                torch.nn.LeakyReLU(0.2),
                convertible_modules['reflection_pad_1'],
                torch.nn.utils.weight_norm(convertible_modules['conv1d_1']),
                torch.nn.LeakyReLU(0.2),
                torch.nn.utils.weight_norm(convertible_modules['conv1d_2']),
            )
        self.__sum__ = convertible_modules['sum']
        self.shortcut = torch.nn.utils.weight_norm(convertible_modules['conv1d_3'])

    def forward(self, x):
        return self.__sum__([self.shortcut(x), self.block(x)])

class VocGan(NeuralNetwork):
    """This class implements vocgan - a popular artificial neural network that maps from speech spectrogram to waveform. It is adopted from https://github.com/rishikksh20/VocGAN ."""

    WAVE_SAMPLING_RATE = 22050
    WAVE_FRAMES_PER_HOP = 256
    WAVE_FRAMES_PER_WINDOW = 1024

    TIMING_CONVENTIONS = {
            "Seconds Per Spectrogram Window": WAVE_FRAMES_PER_WINDOW / WAVE_SAMPLING_RATE,
            "Seconds Per Spectrogram Hop": WAVE_FRAMES_PER_HOP / WAVE_SAMPLING_RATE,
    }

    def __init__(self, is_streamable: bool, input_feature_count: int, output_feature_count: int, residual_layer_count: int, ratios: List[int] = [4, 4, 2, 2, 2, 2], multiplier: int = 256, name:str = "VocGan") -> object:
        """Constructor for this class.
        
        Inputs:
        - is_streamable: Indicates whether this neural network shall be used in streamable or stationary mode.
        - input_feature_count: The number of mel features per time frame.
        - output_feature_count: The number of audio waveform features, usually 1 or 2.
        - residual_layer_count: The number of residual layers to be used in each residual stack. A residual layer allows some of the data to skip a few processing steps.
        - ratios: The input output ratios for the upsample layers.
        - multiplier: The number of timeframes of waveform obtained from one time frame of spectrogram. 
        - name: The name of this neural network.
        """

        # 0. Super
        super(VocGan, self).__init__(is_streamable=is_streamable, name=name)

        # 1. Create a dictionary of stationary modules
        convertible_modules = {
            'reflection_pad_1': torch.nn.ReflectionPad1d(3),
            'conv1d_1': torch.nn.Conv1d(input_feature_count, multiplier * 2, kernel_size=7, stride=1),
            # Upsample 1
            'conv_transpose1d_1': torch.nn.ConvTranspose1d(multiplier * 2, multiplier, kernel_size=ratios[0] * 2, stride=ratios[0], padding=ratios[0] // 2 + ratios[0] % 2, output_padding=ratios[0] % 2),
            # Upsample 2
            'conv_transpose1d_2': torch.nn.ConvTranspose1d((multiplier//2) * 2, multiplier//2, kernel_size=ratios[1] * 2, stride=ratios[1], padding=ratios[1] // 2 + ratios[1] % 2, output_padding=ratios[1] % 2),
            # Upsample 3
            'conv_transpose1d_3': torch.nn.ConvTranspose1d(((multiplier//2)//2) * 2, ((multiplier//2)//2), kernel_size=ratios[2] * 2, stride=ratios[2], padding=ratios[2] // 2 + ratios[2] % 2, output_padding=ratios[2] % 2),
            'conv_transpose1d_4': torch.nn.ConvTranspose1d(input_feature_count, ((multiplier//2)//2), kernel_size=64, stride=32, padding=16, output_padding=0),
            # Upsample 4
            'conv_transpose1d_5': torch.nn.ConvTranspose1d((((multiplier//2)//2)//2) * 2, (((multiplier//2)//2)//2), kernel_size=ratios[3] * 2, stride=ratios[3], padding=ratios[3] // 2 + ratios[3] % 2, output_padding=ratios[3] % 2),
            'conv_transpose1d_6': torch.nn.ConvTranspose1d(input_feature_count, (((multiplier//2)//2)//2), kernel_size=128, stride=64, padding=32, output_padding=0),
            # Upsample 5
            'conv_transpose1d_7': torch.nn.ConvTranspose1d(((((multiplier//2)//2)//2)//2) * 2, ((((multiplier//2)//2)//2)//2), kernel_size=ratios[4] * 2, stride=ratios[4], padding=ratios[4] // 2 + ratios[4] % 2, output_padding=ratios[4] % 2),
            'conv_transpose1d_8': torch.nn.ConvTranspose1d(input_feature_count, ((((multiplier//2)//2)//2)//2), kernel_size=256, stride=128, padding=64, output_padding=0),
            # Upsample 6
            'conv_transpose1d_9': torch.nn.ConvTranspose1d((((((multiplier//2)//2)//2)//2)//2) * 2, (((((multiplier//2)//2)//2)//2)//2), kernel_size=ratios[5] * 2, stride=ratios[5], padding=ratios[5] // 2 + ratios[5] % 2, output_padding=ratios[5] % 2),
            'conv_transpose1d_10': torch.nn.ConvTranspose1d(input_feature_count, (((((multiplier//2)//2)//2)//2)//2), kernel_size=512, stride=256, padding=128, output_padding=0),
            'reflection_pad_2': torch.nn.ReflectionPad1d(3),
            'conv1d_2': torch.nn.Conv1d((((((multiplier//2)//2)//2)//2)//2), output_feature_count, kernel_size=7, stride=1),
            # Sum
            'sum_1': stationary.Sum(),
            'sum_2': stationary.Sum(),
            'sum_3': stationary.Sum(),
            'sum_4': stationary.Sum()
            }
        
        # 2. Convert them to streamable modules
        if is_streamable: convertible_modules = NeuralNetwork.__to_streamable__(modules=convertible_modules)
        
        # 3. Save the convertible modules for later computations 
        # Start layer
        self.start = torch.nn.Sequential(
            convertible_modules['reflection_pad_1'],
            torch.nn.utils.weight_norm(convertible_modules['conv1d_1'])
        )

        # Upsample 1
        self.upsample_1 = torch.nn.Sequential(
            torch.nn.LeakyReLU(0.2),
            torch.nn.utils.weight_norm(convertible_modules['conv_transpose1d_1'])
        )
        self.res_stack_1 = torch.nn.Sequential(*[ResStack(is_streamable=is_streamable, feature_count=multiplier, dilation=3 ** j) for j in range(residual_layer_count)])

        # Upsample 2
        multiplier = multiplier // 2
        self.upsample_2 = torch.nn.Sequential(
            torch.nn.LeakyReLU(0.2),
            torch.nn.utils.weight_norm(convertible_modules['conv_transpose1d_2'])
        )
        self.res_stack_2 = torch.nn.Sequential(*[ResStack(is_streamable=is_streamable, feature_count=multiplier, dilation=3 ** j) for j in range(residual_layer_count)])

        # Upsample 3
        multiplier = multiplier // 2
        self.upsample_3 = torch.nn.Sequential(
            torch.nn.LeakyReLU(0.2),
            torch.nn.utils.weight_norm(convertible_modules['conv_transpose1d_3'])
        )

        self.skip_upsample_1 = torch.nn.utils.weight_norm(convertible_modules['conv_transpose1d_4'])
        self.res_stack_3 = torch.nn.Sequential(*[ResStack(is_streamable=is_streamable, feature_count=multiplier, dilation=3 ** j) for j in range(residual_layer_count)])

        # Upsample 4
        multiplier = multiplier // 2
        self.upsample_4 = torch.nn.Sequential(
            torch.nn.LeakyReLU(0.2),
            torch.nn.utils.weight_norm(convertible_modules['conv_transpose1d_5'])
        )

        self.skip_upsample_2 = torch.nn.utils.weight_norm(convertible_modules['conv_transpose1d_6'])
        self.res_stack_4 = torch.nn.Sequential(*[ResStack(is_streamable=is_streamable, feature_count=multiplier, dilation=3 ** j) for j in range(residual_layer_count)])

        # Upsample 5
        multiplier = multiplier // 2
        self.upsample_5 = torch.nn.Sequential(
            torch.nn.LeakyReLU(0.2),
            torch.nn.utils.weight_norm(convertible_modules['conv_transpose1d_7'])
        )

        self.skip_upsample_3 = torch.nn.utils.weight_norm(convertible_modules['conv_transpose1d_8'])
        self.res_stack_5 = torch.nn.Sequential(*[ResStack(is_streamable=is_streamable, feature_count=multiplier, dilation=3 ** j) for j in range(residual_layer_count)])

        # Upsample 6
        multiplier = multiplier // 2
        self.upsample_6 = torch.nn.Sequential(
            torch.nn.LeakyReLU(0.2),
            torch.nn.utils.weight_norm(convertible_modules['conv_transpose1d_9'])
        )

        self.skip_upsample_4 = torch.nn.utils.weight_norm(convertible_modules['conv_transpose1d_10'])
        self.res_stack_6 = torch.nn.Sequential(*[ResStack(is_streamable=is_streamable, feature_count=multiplier, dilation=3 ** j) for j in range(residual_layer_count)])

        # Output layer
        self.out = torch.nn.Sequential(
            torch.nn.LeakyReLU(0.2),
            convertible_modules['reflection_pad_2'],
            torch.nn.utils.weight_norm(convertible_modules['conv1d_2']),
            torch.nn.Tanh(),
        )

        self.apply(VocGan.weights_init)

        # Save sums for later
        self.__sum_1__ = convertible_modules['sum_1']
        self.__sum_2__ = convertible_modules['sum_2']
        self.__sum_3__ = convertible_modules['sum_3']
        self.__sum_4__ = convertible_modules['sum_4']

    @staticmethod
    def weights_init(m):
        """Weight initialization"""
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, mel):
        mel = (mel + 5.0) / 5.0 # roughly normalize spectrogram
        # Mel Shape [B, num_mels, T] -> torch.Size([3, 80, 10])
        x = self.start(mel)  # [B, dim*2, T] -> torch.Size([3, 512, 10])

        x = self.upsample_1(x)
        x = self.res_stack_1(x)  # [B, dim, T*4] -> torch.Size([3, 256, 40])

        x = self.upsample_2(x)
        x = self.res_stack_2(x)  # [B, dim/2, T*16] -> torch.Size([3, 128, 160])
        
        x = self.upsample_3(x)
        x = self.__sum_1__([x, self.skip_upsample_1(mel)])
        x = self.res_stack_3(x)  # [B, dim/4, T*32] -> torch.Size([3, 64, 320])
        
        x = self.upsample_4(x)
        x = self.__sum_2__([x, self.skip_upsample_2(mel)])
        x = self.res_stack_4(x)  # [B, dim/8, T*64] -> torch.Size([3, 32, 640])
        
        x = self.upsample_5(x)
        x = self.__sum_3__([x, self.skip_upsample_3(mel)])
        x = self.res_stack_5(x)  # [B, dim/16, T*128] -> torch.Size([3, 16, 1280])
        
        x = self.upsample_6(x)
        x = self.__sum_4__([x, self.skip_upsample_4(mel)])
        x = self.res_stack_6(x)  # [B, dim/32, T*256] -> torch.Size([3, 8, 2560])

        out = self.out(x)  # [B, 1, T*256] -> torch.Size([3, 1, 2560])

        return out

    def load(is_streamable: bool, path: str = None) -> NeuralNetwork:
        """Loads the neural network parameters
        
        Inputs:
        - is_streamable: Indicates whether the instance shall be used for streamable or not.
        - path: The path to a pretrained model. If None then vctk_pretrained_model_3180.pt is used.
        
        Outputs:
        - self: The instance."""

        # Load parameters
        model_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'parameters', "vctk_pretrained_model_3180.pt"))
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        hp = hparams.load_hparam_str(checkpoint['hp_str'])

        # Create model
        neural_network = VocGan(is_streamable=is_streamable, input_feature_count=hp.audio.n_mel_channels, output_feature_count=hp.model.out_channels, 
            residual_layer_count=hp.model.n_residual_layers, ratios=hp.model.generator_ratio, multiplier=hp.model.mult)
        neural_network.load_state_dict(checkpoint['model_g'])
        neural_network.remove_weight_norm()
        # Outputs
        return neural_network

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    @staticmethod
    def waveform_to_mel_spectrogram(waveform: Union[torch.Tensor, np.array], original_sampling_rate: int, mel_feature_count: int = 80, min_frequency: float = 0.0, max_frequency: float = 8000.0) -> torch.Tensor:
        """Converts waveform audio to mel spectrogram.
        
        Inputs:
        - waveform: The audio waveform signal of shape [instance count, time frame count] or [time frame count].
        - original_sampling_rate: the sampling rate of the waveform.
        - mel_feature_count: Optional argument. The number of mel features the final spectrogram should have.
        - min_frequency: Optional argument. The minimum frequency of the mel spectrogram.
        - max_frequency: Optional argument. The maximum frequency of the mel spectrogram.
        
        Outputs:
        - mel_spectrogram: The mel spectrogram. It adheres to VocGan.TIMING_CONVENTIONS and is of shape [instance count, mel feature count, time frame count] or [mel feature count, time frame count]."""

        # Input management
        if not type(waveform) == type(torch.Tensor): waveform = torch.Tensor(waveform)
        if len(waveform.shape) == 1: waveform = waveform.unsqueeze(0)
        waveform = waveform.to(torch.float32)
        for i in range(waveform.shape[0]): waveform[i] = waveform[i]/ torch.max(torch.abs(waveform[i]))

        # Converting step parameters according to current sampling rate
        hop_length= (int)(VocGan.WAVE_FRAMES_PER_HOP/VocGan.WAVE_SAMPLING_RATE*original_sampling_rate)
        win_length= (int)(VocGan.WAVE_FRAMES_PER_WINDOW/VocGan.WAVE_SAMPLING_RATE*original_sampling_rate)
        
        # Creating Mel Spectrogram
        fourier = stft.TacotronSTFT(filter_length=win_length, hop_length=hop_length, win_length=win_length,
            n_mel_channels=mel_feature_count, sampling_rate=original_sampling_rate, mel_fmin=min_frequency, mel_fmax=max_frequency)
        mel_spectrogram = fourier.mel_spectrogram(y=waveform).squeeze()

        # Outputs
        return mel_spectrogram

    @staticmethod 
    def plot(mel_spectrogram: np.array, waveform_stationary: torch.Tensor, waveform_streamable_slices: List[torch.Tensor] = None, slice_processing_times: List[float] = None) -> None:
        """Plots the inputs and outputs of mel_spectrogram_to_waveform. 
        
        Inputs:
        - mel_spectrogram: The spectrogram to be plotted. Shape = [mel feature count, time point count]. 
        - waveform_stationary: The waveform to be plotted. Shape = [time point count]. 
        - waveform_streamable_slices: Optional argument. Provides slices of the second waveform to be plotted.
        - slice_processing_times: Optional argument. If waveform_streamable_slices is provided then a another plot is shown that plots for every slice its processing time. Times are assumed to be in seconds."""
        
        # Verify arguments
        if type(slice_processing_times) != type(None):
            if type(waveform_streamable_slices) == type(None):
                raise ValueError("If the argument slice_processing_times is provided, then waveform_streamable_slices has to be provided.")
            
        # Cast to numpy arrays
        waveform_stationary = waveform_stationary.detach().numpy()
        if waveform_streamable_slices != None: waveform_streamable_slices = [slice.detach().numpy() for slice in waveform_streamable_slices]

        # Count plots
        plot_count = 2 + (1 if type(waveform_streamable_slices) != type(None) else 0) + (1 if type(slice_processing_times) != type(None) else 0) 

        # Time ticks
        time_frame_count = mel_spectrogram.shape[-1]
        total_seconds = (time_frame_count-1) * VocGan.TIMING_CONVENTIONS["Seconds Per Spectrogram Hop"] + VocGan.TIMING_CONVENTIONS["Seconds Per Spectrogram Window"] 
        
        # Plots
        # Font
        plt.rcParams['font.sans-serif'] = "Times New Roman"
        plt.rcParams['font.family'] = "sans-serif"

        # Spectrogram
        plt.figure()
        plt.subplot(plot_count,1,1)
        plt.imshow(np.flipud(mel_spectrogram), aspect='auto')
        time_frame_count = mel_spectrogram.shape[-1]
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
        plt.ylabel("Mel Bin")
        plt.title("Spectrogram")
            
        # Standard output
        plt.subplot(plot_count,1,2)
        plt.plot(waveform_stationary)
        plt.title("Standard Waveform")
        time_frame_count = waveform_stationary.shape[-1]
        plt.xlim(0,time_frame_count)
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
        plt.ylabel("Amplitude")

        # streamable output
        if type(waveform_streamable_slices) != type(None):
            plt.subplot(plot_count,1,3)
            plt.plot(np.concatenate(waveform_streamable_slices, axis=-1))
            plt.xticks(ticks=[])
            plt.yticks(ticks=[])
            plt.title("streamable Waveform")
            plt.xlim(0,time_frame_count)
            plt.ylabel("Amplitude")
        
            # Processing times
            if type(slice_processing_times) != type(None):
                plt.subplot(plot_count,1,4)
                slice_starts = np.cumsum([0] + [slice.shape[-1] for slice in waveform_streamable_slices[:-1]]) # Now the first number is the first slice's starting index (0), last number is last slice's starting index
                slice_stops = np.cumsum([slice.shape[-1] for slice in waveform_streamable_slices]) # Now the first number is the stop index (exclusive) of the first slice, last number is stop index of last slice
                slice_durations = np.array([slice.shape[-1]/VocGan.WAVE_SAMPLING_RATE for slice in waveform_streamable_slices])
                efficiencies = [slice_processing_times[i] / slice_durations[i] if slice_durations[i] > 0 else 0 for i in range(len(slice_durations))]
                plt.bar(x=slice_starts+(slice_stops-slice_starts)/2, height=efficiencies, width=slice_stops-slice_starts, fill=False)
                
                plt.axhline(y = 1, color = 'r', linestyle = '-')
                plt.xticks(ticks=[])
                plt.xlim(0,time_frame_count)
                plt.ylabel("Efficiency")
                plt.title("streamable Efficiency")

                # Text box
                textstr = ',   '.join((
                    '   Efficiency = Processing Time / Audio Duration',
                    r'5 percent trimmed average Audio Duration = %.4f sec' % (scipy.stats.trim_mean(slice_durations, 0.05)),
                    r'5 percent trimmed average Processing Time = %.4f sec' % (scipy.stats.trim_mean(slice_processing_times, 0.05))))

                props = dict(boxstyle='round', facecolor='white', alpha=0.5)

                # place a text box in upper left in axes coords
                plt.text(0, 0.9*plt.ylim()[1], textstr, fontsize=9,
                        verticalalignment='top', bbox=props)

        # Timeline for bottom plot
        time_frame_count = waveform_stationary.shape[-1]
        tick_locations = np.arange(stop=time_frame_count, step=VocGan.WAVE_SAMPLING_RATE)
        tick_labels = np.arange(stop=total_seconds, step=1.0)
        plt.xticks(ticks=tick_locations, labels=tick_labels)
        plt.xlim(0,time_frame_count)
        plt.xlabel("Signal Time (seconds)")
        
        plt.show()

    def mel_spectrogram_to_waveform(self, mel_spectrogram: torch.Tensor, is_final_slice: bool = False) -> Tuple[torch.Tensor, float]:
        """Converts a mel spectrogram to a waveform. 
        
        Inputs:
        - mel_spectrogram: Mel spectrogram of shape [instance count, mel feature count, time point count] or [mel feature count, time point count]. Each spectrogram timeframe is assumed to adhere to VocGan.TIMING_CONVENTIONS.
        - is_final_slice: Indicates whether this is the final slice. If self.is_streamable == True then is_final_slice needs to be specfied.
        
        Outputs:
        - waveform: The time domain signal obtained from x. If the mel_spectrogram had multiple instance, waveform will have shape [instance count, time frame count], else it will have shape [time frame count].
        - processing_time: The time it took to convert the mel_spectrogram to the waveform in seconds."""
        
        # Some constants
        mel_feature_count = mel_spectrogram.shape[-2]
        tick = time.time()

        # Predict
        with torch.no_grad():
            # Unsqueeze along instance dimension if we have a single instance
            if len(mel_spectrogram.shape) == 2:
                mel_spectrogram = mel_spectrogram.unsqueeze(0) # Now shape == [instance count, feature count, time frame count]
            
            # Predict
            if is_final_slice: streamable.Module.final_slice()
            
            # pad input mel to cut artifact, see https://github.com/seungwonpark/melgan/issues/8
            if (not self.__is_streamable__) or is_final_slice: 
                padding = torch.full((1, mel_feature_count, 10), -11.5129)
                mel_spectrogram = torch.cat((mel_spectrogram, padding), dim=2) 

            waveform = self.forward(mel_spectrogram)
            
            if is_final_slice: streamable.Module.close_stream()

            # Post process
            # Related to the artifact mentioned above
            if (not self.__is_streamable__) or is_final_slice: 
                waveform = waveform[:,:,:-(VocGan.WAVE_FRAMES_PER_HOP*10)]
            waveform = waveform.squeeze()

            processing_time = time.time() - tick # In seconds

            return waveform, processing_time

    @staticmethod
    def slice_duration_to_frame_count(spectrogram_time_frame_count, target_seconds_per_slice: float) -> Tuple[float, float, int]:
        """Helps to configure the slice generation by providing time frame count and duration for slices of a spectrogram. 
        Assumes that the spectrogram adheres to VocGan.TIMING_CONVENTIONS.
        
        Inputs:
        - spectrogram_time_frame_count: The total number of time frames in the spectrogram.
        - target_seconds_per_slice: The desired amount of new seconds per slice. Beware that a single time frame covers about 46ms of audio but due to overlap
            the next time frame will only add about 12ms of new audio (see VocGan.TIMING_CONVENTIONS). Here, the parameter target_seconds_per_slice
            assumes a slice for when the stream is running. Hence it disregardes overlap with the previous slide and only considers the new duration that this slice provides.
            Note that due to rastorization the actual duration might differ slightly (see output).
        
        Outputs:
        - time_frame_count_per_slice: The number of spectrogram time frames that should be used per slice.
        - actual_seconds_per_slice: The actual duration spanned by the slice in seconds.
        - slice_count: The number of slices that can be obtained with the time_frame_count. Note that all slices are assumed to have the here computed time frame count, expect for the last slice which may be shorter."""

        # Compute number of time frames
        b = VocGan.TIMING_CONVENTIONS["Seconds Per Spectrogram Hop"] 
        time_frames_per_slice = (int)(np.round((target_seconds_per_slice)/b)) 
        
        # Actual duration of slice
        actual_seconds_per_slice = (time_frames_per_slice) * b 

        # Number of slices
        slice_count = math.ceil(spectrogram_time_frame_count/time_frames_per_slice)

        # Outputs
        return time_frames_per_slice, actual_seconds_per_slice, slice_count

    @staticmethod
    def slice_generator(mel_spectrogram: torch.Tensor, time_frames_per_slice: float) -> Tuple[torch.Tensor, bool]:
        """Generates slices of the input tensor x such that they can be fed to the convert function.
        
        Inputs:
        - mel_spectrogram: The input mel spectrogram to convert with all time frames. Shape == [mel feature count, time frame count]
        - time_frame_count_per_slice: The slice size in time frames. 
        
        Outputs:
        - x_i: A slice of the input x. Time frame count is always equal to slice_size except for the last slice whose lenght k is 0 < k <= slice_size.
        - is_final_slice: Indicates whether this slice is the final one."""

        # Step management
        time_frame_count = mel_spectrogram.shape[-1]
        slice_count = math.ceil(time_frame_count/ time_frames_per_slice)

        # Iterator
        for i in range(slice_count):
            x_i = mel_spectrogram[:,i*time_frames_per_slice:(i+1)*time_frames_per_slice]
            is_final_slice = i == slice_count - 1
            # Outputs
            yield (x_i, is_final_slice)

    @staticmethod
    def save(waveform: torch.Tensor, file_path: str) -> None:
        """Saves the audio to file. 
        
        Calling Instruction:
        - Call this method only on the entire sequence, rather than individual snippets. This method scales the audio such that its maximum absolute value is equal to the maximum absolute value of int16.
        
        Inputs:
        - audio: The audio sequence to be saved. Shape = [time frame count]. Assumed to be sampled at self.SAMPLING_RATE.
        - file_path: The path to the target file including file name and .wav extension.

        Outputs:
        - None
        """
        # Cast to numpy
        waveform = waveform.detach().numpy()

        # Scale
        amplitude = np.iinfo(np.int16).max
        waveform = (amplitude*(waveform/np.max(np.abs(waveform)))).astype(np.int16)
        
        # Save
        wavfile.write(filename=file_path, rate=VocGan.WAVE_SAMPLING_RATE, data=waveform)
    
class SpeechAutoEncoder(NeuralNetwork):
    """This neural network can be used to auto encode speech in the form of spectrograms."""

    def __init__(self, input_feature_count: int, x_alphabet: torch.Tensor, is_streamable: bool = False, name: str = "SpeechAutoEncoder") -> object:
        """Constructor for this class.
        
        Inputs:
        - input_feature_count: number of input features. The number of output features will be the same.
        - x_alphabet: a matrix that contains alphabet vectors that the network learns to select for its latent representation. Shape == [instance count, feature count]
        - is_streamable: indicates whether this neural network shall be used in streamable or stationary mode.
        - name: the name of this neural network
        """
    
        # Following the super class instructions for initialization
        # 0. Super
        super(SpeechAutoEncoder, self).__init__(is_streamable=is_streamable, name=name)

        # Constants
        latent_feature_count = x_alphabet.size()[-1]
        self.__x_value__ = copy.deepcopy(x_alphabet) # Shape == [time frames of value, features of value]
        self.__x_key__ = copy.deepcopy(x_alphabet).permute(1,0) # Shape == [features of value, time frames of value]

        # 1. Create a dictionary of stationary modules
        convertible_modules = {
            # Encoder
            'content_encoder_linear': torch.nn.Linear(in_features=input_feature_count, out_features=latent_feature_count),
            'content_encoder_convolutional_1': torch.nn.Conv1d(latent_feature_count, latent_feature_count, kernel_size=8, dilation=8),
            'content_encoder_convolutional_2': torch.nn.Conv1d(latent_feature_count, latent_feature_count, kernel_size=4, dilation=4),
            'content_encoder_convolutional_3': torch.nn.Conv1d(latent_feature_count, latent_feature_count, kernel_size=2, dilation=2),
            'content_encoder_convolutional_4': torch.nn.Conv1d(latent_feature_count, latent_feature_count, kernel_size=8, dilation=8),
            'content_encoder_pad_1': torch.nn.ReplicationPad1d(padding=[(8-1)*8//2,(8-1)*8//2]),
            'content_encoder_pad_2': torch.nn.ReplicationPad1d(padding=[(4-1)*4//2,(4-1)*4//2]),
            'content_encoder_pad_3': torch.nn.ReplicationPad1d(padding=[(2-1)*2//2,(2-1)*2//2]),
            'content_encoder_pad_4': torch.nn.ReplicationPad1d(padding=[(8-1)*8//2,(8-1)*8//2]),
            'content_encoder_sum': stationary.Sum(),
            
            # Style
            'style_encoder_linear': torch.nn.Linear(in_features=input_feature_count, out_features=latent_feature_count),
            'style_encoder_convolutional_1': torch.nn.Conv1d(latent_feature_count, latent_feature_count, kernel_size=8, dilation=8),
            'style_encoder_convolutional_2': torch.nn.Conv1d(latent_feature_count, latent_feature_count, kernel_size=4, dilation=4),
            'style_encoder_convolutional_3': torch.nn.Conv1d(latent_feature_count, latent_feature_count, kernel_size=2, dilation=2),
            'style_encoder_convolutional_4': torch.nn.Conv1d(latent_feature_count, latent_feature_count, kernel_size=8, dilation=8),
            'style_encoder_pad_1': torch.nn.ReplicationPad1d(padding=[(8-1)*8//2,(8-1)*8//2]),
            'style_encoder_pad_2': torch.nn.ReplicationPad1d(padding=[(4-1)*4//2,(4-1)*4//2]),
            'style_encoder_pad_3': torch.nn.ReplicationPad1d(padding=[(2-1)*2//2,(2-1)*2//2]),
            'style_encoder_pad_4': torch.nn.ReplicationPad1d(padding=[(8-1)*8//2,(8-1)*8//2]),
            'style_encoder_sum': stationary.Sum(),
            'style_rnn_1': torch.nn.LSTM(latent_feature_count, latent_feature_count, num_layers=1, batch_first=True),

            # Decoder
            'decoder_convolutional_1': torch.nn.Conv1d(latent_feature_count, latent_feature_count, kernel_size=2, dilation=2),
            'decoder_convolutional_2': torch.nn.Conv1d(latent_feature_count, latent_feature_count, kernel_size=4, dilation=4),
            'decoder_convolutional_3': torch.nn.Conv1d(latent_feature_count, latent_feature_count, kernel_size=8, dilation=8),
            'decoder_convolutional_4': torch.nn.Conv1d(latent_feature_count, latent_feature_count, kernel_size=8, dilation=8),
            'decoder_pad_1': torch.nn.ReplicationPad1d(padding=[(2-1)*2,0]),
            'decoder_pad_2': torch.nn.ReplicationPad1d(padding=[(4-1)*4,0]),
            'decoder_pad_3': torch.nn.ReplicationPad1d(padding=[(8-1)*8,0]),
            'decoder_pad_4': torch.nn.ReplicationPad1d(padding=[(8-1)*8,0]),
            'decoder_sum': stationary.Sum(),
            'decoder_linear': torch.nn.Linear(in_features=latent_feature_count, out_features=input_feature_count)
            }
        
        # 2. Convert them to streamable modules
        if is_streamable: convertible_modules = NeuralNetwork.__to_streamable__(modules=convertible_modules)
        
        # 3 Save the convertible modules for later computations 
        # 3.1 Encoder
        self.content_encoder_long_path = torch.nn.Sequential(
            Transpose(),
            convertible_modules['content_encoder_pad_1'],
            convertible_modules['content_encoder_convolutional_1'], torch.nn.ReLU(),
            convertible_modules['content_encoder_pad_2'],
            convertible_modules['content_encoder_convolutional_2'], torch.nn.ReLU(),
            convertible_modules['content_encoder_pad_3'],
            convertible_modules['content_encoder_convolutional_3'],
            Transpose()
        )

        self.content_encoder_linear = convertible_modules['content_encoder_linear']
        self.content_encoder_short_cut = torch.nn.Sequential(Transpose(), convertible_modules['content_encoder_pad_4'], convertible_modules['content_encoder_convolutional_4'], Transpose()) 
        self.content_encoder_sum = convertible_modules['content_encoder_sum']

        # Style
        self.style_encoder_long_path = torch.nn.Sequential(
            Transpose(),
            convertible_modules['style_encoder_pad_1'],
            convertible_modules['style_encoder_convolutional_1'], torch.nn.ReLU(),
            convertible_modules['style_encoder_pad_2'],
            convertible_modules['style_encoder_convolutional_2'], torch.nn.ReLU(),
            convertible_modules['style_encoder_pad_3'],
            convertible_modules['style_encoder_convolutional_3'],
            Transpose()
        )

        self.style_encoder_linear = convertible_modules['style_encoder_linear']
        self.style_encoder_short_cut = torch.nn.Sequential(Transpose(), convertible_modules['style_encoder_pad_4'], convertible_modules['style_encoder_convolutional_4'], Transpose()) 
        self.style_encoder_sum = convertible_modules['style_encoder_sum']

        self.style_rnn = convertible_modules['style_rnn_1']

        # 3.2 Decoder
        self.decoder_long_path = torch.nn.Sequential(
            Transpose(),
            convertible_modules['decoder_pad_1'],
            convertible_modules['decoder_convolutional_1'], torch.nn.ReLU(),
            convertible_modules['decoder_pad_2'],
            convertible_modules['decoder_convolutional_2'], torch.nn.ReLU(),
            convertible_modules['decoder_pad_3'],
            convertible_modules['decoder_convolutional_3'],
            Transpose()
        )

        self.decoder_short_cut = torch.nn.Sequential(Transpose(), convertible_modules['decoder_pad_4'], convertible_modules['decoder_convolutional_4'],Transpose()) 
        self.decoder_sum = convertible_modules['decoder_sum']
        self.decoder_linear = convertible_modules['decoder_linear']


    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Executes forward calculation for this neural network. It uses two inputs, a content and a style input.
        These are both speech spectrograms of the same voice yet with different words being spoken.
        The network has a content path that goes via a convolutional encoder to an attention module that selects 
        vectors from x_alphabet provided by the constructor. It also has a style path that goes via the same convolutional encoder to a recurrent module.
        The recurrent module accumulated the style informtion. Its final output vector is multiplied elementwise with each phoneme vector 
        seleced by the attention module of the content path. These latent vectors are then passed through a convolutional decoder.
        The number of time frames is equal for input, latent and output.
        
        Inputs:
        - x: List of two tensors [x_content, x_style]. Each of shape [instances per batch, time frames per instance, input feature count].
        
        Outputs:
        - y_hat: tensor of shape [instances per batch, time frames per content instance, input_feature_count]."""
        
        # Input validity
        assert type(x) == type([]), f"Expected input x to be of type list, received {type(x)}."
        assert len(x) == 2, f"Expected input x to be a list with two elements, i.e. x_content and x_style. Received list of length {len(x)}."
        assert x[0].size()[0] == x[1].size()[0], f"Expected x_content and x_style to have the same number of instances per batch. Received {x[0].size()[0]} for x_content and {x[1].size()[0]} for x_style."
        assert len(x[0].size()) == 3 and len(x[1].size()) == 3, f"Expected x_content and x_style to have three dimensions. Received shape {x[0].size()} for x_content and {x[1].size()} for x_style."
        assert x[0].size()[-1] == x[1].size()[-1], f"Expected x_content and x_style to have the same number of features. Received {x[0].size()[-1]} for x_content and {x[1].size()[-1]} for x_style."
        
        # Unpack inputs
        x_content, x_style = x[0], x[1]

        # Encode
        x_content = self.content_encoder_linear(x_content); x_style = self.style_encoder_linear(x_style)
        x_query = self.content_encoder_sum([self.content_encoder_long_path(x_content), self.content_encoder_short_cut(x_content)]) # Shape == [instances per batch, time frames per instance, latent feature count]
        x_style = self.style_encoder_sum(  [self.style_encoder_long_path(x_style),     self.style_encoder_short_cut(x_style)]) # Shape == [instances per batch, time frames per instance, latent feature count]
        x_style, _ = self.style_rnn(x_style) 
        x_style = x_style[:,-1,:].unsqueeze(1) # Final output of rnn. Shape == [instances per batch, 1 time frame, latent feature count]

        # Form attention matrix
        A = x_query.matmul(self.__x_key__) # Shape == [instances per batch, time frames per query, time frames of x_key]
        A = A / (torch.std(A, dim=-1).unsqueeze(-1)+ 1e-5) # Scale for stability. The epsilon avoids division by zero
        A = F.softmax(A, dim=-1) # Shape == [instances per batch, time frames per query, time frames of key]
        
        # Select from x_value by attention (remember that x_value and x_key have the same time frame count)
        x_latent = A.matmul(self.__x_value__) # Shape == [instances per batch, time frames per query, latent feature count]
        
        # Combine style and content
        x_latent = x_latent * x_style

        # Decode
        y_hat = self.decoder_sum([self.decoder_long_path(x_latent), self.decoder_short_cut(x_latent)])
        y_hat = self.decoder_linear(y_hat)

        # Outputs
        return y_hat, A