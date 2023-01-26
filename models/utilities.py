import torch
from typing import List, Union, Tuple

def reshape_by_time_frame_count(x: torch.Tensor, time_frames_per_instance: int) -> torch.Tensor:
    """Splits x into a batches of instances.
    
    Inputs:
    - x: tensor of shape [time frame count, channel count].
    - time_frames_per_instance: number of timeframes per instance.
    
    Outputs:
    - x: tensor of shape [instances_per_batch, time_frames_per_instance, channel count]. Trailing time frames of the original x that did not fit into this shape are excluded."""

    # Validate input
    assert type(x) == torch.Tensor, f"Input x must be of type torch.Tensor, not {type(x)}."
    assert len(x.size()) == 2, f"Input x expected to have two axes, first for time frames, second for channels. Received x of shape {x.size()}."

    # Remove surplus time frames
    time_frame_count = x.size()[0]
    surplus_time_frames = time_frame_count %  time_frames_per_instance
    time_frame_count -= surplus_time_frames
    x = x[:time_frame_count, :]
        
    # Reshape x
    channel_count = x.size()[1]
    instances_per_batch = time_frame_count // time_frames_per_instance
    x = x.reshape(shape=(instances_per_batch, time_frames_per_instance, channel_count))

    # Outputs
    return x

def reshape_by_label(x: torch.Tensor, labels: List[str], pause_string: str, y: torch.Tensor = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Reshapes the matrices x, y of shape [instance count * time frame count, feature count] into tensors
    of shape [instance count, time frame count, feature count]. Note that initially x and y do not have to 
    have exactly instance count * time frame count rows since the labels input is used to identify instance boundaries.
    Padding is applied to the right of every instance if needed, but not on the left.
    For efficiency it is recommended to use this function on x and y simultaneously rather than separately.

    Inputs:
    - x: The input to a neural network. Shape == [instance count * time frame count, x feature count]
    - labels: List labels. Every jump from pause_string to non-pause string is used as a splitting point for x and y.
    - pause_string: A string that separates instances.
    - y: The target of the neural network. Shape == [instance count * time frame count, y feature count]
    
    Assumptions:
    - Inputs x, y and labels are assumed to have the same number entries along the first axis.
    - At least one time frame is assumed to exist.

    Outputs:
    - x_new: The reshaped x. Shape == [instance count, time frame count, x feature count].
    - y_new: Returned if y was provided as input, The reshaped y. Shape == [instance count, time frame count, y feature count]."""

    # Input validity
    assert type(x) == torch.Tensor, f"Expected x to have type torch.Tensor, received {type(x)}"
    if type(y) != type(None): assert type(y) == torch.Tensor, f"Expected y to have type torch.Tensor, received {type(y)}"
    assert type(labels) == type([]), f"Expected labels to have type list, received {type(labels)}"
    if type(y) != type(None): assert (len(x) == len(y) and len(y) == len(labels)), f"Inputs must have same length. Received x of shape {x.size()}, y of shape {y.size()} and labels of length {len(labels)}"
    assert len(labels) > 0, f"Expected there to be at least one time frame, found 0"

    # Overwrite pause characters with their preceding character
    for l in range(len(labels)-1): 
        if labels[l+1] == pause_string: labels[l+1] = labels[l]

    # Initialize new arrays with length equal to upper bound of possible length
    x_new = [None] * len(x)
    if type(y) != type(None): y_new = [None] * len(y)

    # Fill the new arrays
    i = 0 # index for the new arrays
    s = 0; f = s + 1 # start and finish indices for current snippet in labels
    longest_sequence_length = 0
    for label in labels[1:] + ['This is a dummt label used to ensure the last snippet gets copied']:
        label_previous = labels[f-1]
        if label == label_previous: # We are still inside the current snippet
            f += 1
            if f-s >= longest_sequence_length: longest_sequence_length = f-s
        if label != label_previous or f == len(labels) + 1: # We finished the current snippet
            x_new[i] = x[s:f,:]
            if type(y) != type(None): y_new[i] = y[s:f,:] # Copy snippet
            i += 1; s = f; f += 1 # Set indices
          
    # Pad x_new, y_new
    for j in range(i):
        x_zeros = torch.zeros([longest_sequence_length - len(x_new[j]), x_new[j].shape[1]])
        x_new[j] = torch.cat([x_new[j], x_zeros], axis=0).unsqueeze(0)

        if type(y) != type(None): y_zeros = torch.zeros([longest_sequence_length - len(y_new[j]), y_new[j].shape[1]])
        if type(y) != type(None): y_new[j] = torch.cat([y_new[j], y_zeros], axis=0).unsqueeze(0)
      
    # Concatenate
    x_new = torch.cat(x_new[:i], axis=0)
    if type(y) != type(None): y_new = torch.cat(y_new[:i], axis=0)
  
    # Output
    if type(y) != type(None): return x_new, y_new
    else: return x_new

def undo_reshape_by_label(y: torch.Tensor, labels: List[str], pause_string: str, x: torch.Tensor = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Undos the operation of reshape_by_label.
    
    Inputs:
    - y: Output of a neural network. Shape == [instance count, time frame count, y feature count].
    - labels: List of strings labelling each time frame in the original x, i.e. the one used as input to reshape_by_label.
    - pause_string: A string that separates instances.
    - x: Input to a neural network. Shape == [instance count, time frame count, x feature count].
    
    Assumptions:
    - Inputs x, y are assumed to have the same number entries along the initial 2 axes.
    - At least one time frame is assumed to exist in labels.
    - Assumes that reshape_by_label padded instances on the right, not the left.

    Outputs:
    - y_new: Reshaped to [length of labels, y feature count].
    - x_new: Returned if an x was provided as input. Reshaped to [length of labels, x feature count].
    
    """

    # Input validity
    assert type(y) == torch.Tensor, f"Expected y to have type torch.Tensor, received {type(y)}."
    assert len(y.size()) == 3, f"Expected y to have shape [instance count, time frame count, y feature count], received shape {y.size()}."

    if type(x) != type(None):
        assert type(x) == torch.Tensor, f"Expected x to have type torch.Tensor, received {type(x)}."
        assert len(x.size()) == 3, f"Expected x to have shape [instance count, time frame count, x feature count], received shape {x.size()}."
        assert x.size()[0] == y.size()[0], f"Expected x and y to have the same number of instances (first axis), received for x {x.size()[0]} and for y {y.size()[0]}"
  
    # Replace pause_string
    for k in range(len(labels)-1):
        if labels[k+1] == pause_string: labels[k+1] = labels[k]

    # Undo paddings
    instance_count = y.size()[0]
    ys = [None] * instance_count
    if type(x) != type(None): xs = [None] * instance_count
    f = 0  # Index in labels for the last label that is equal to the current label
    for i in range(instance_count):
        # Determine length 
        current_label = labels[f]
        s = 0 # Index in y (and x) for the last time frame whose label is equal to the current label 
        while f < len(labels) and labels[f] == current_label: 
            f += 1
            s += 1
    
        # Extract current section without padding
        ys[i] = y[i,:s,:]
        if type(x) != type(None): xs[i] = x[i,:s,:]

    # Reshape
    y_new = torch.cat(ys, dim=0)
    if type(x) != type(None): x_new = torch.cat(xs, dim=0)

    # Outputs
    if type(x) != type(None): return x_new, y_new
    else: return y_new

def stack_x(x: torch.Tensor, shift_count: int, shift_step_size: int) -> torch.Tensor:
    """Stacks x by concatenating past time frames to each current time frame along the channel axis. 
    Zero padding is applied to the left of each instance to ensure the number of time frames stay constant.
    
    Inputs:
    - x: input to the neural network. Shape == [instances per batch, time frames per instance, channel count].
    - shift_count: Integer at least 1. The number of time frames in the stack. If set to 1 then just the current time frame is in the stack. 
        If set to 2 then one time frame shift_step_size timeframes into the past is concatenated with the current time 
        frame along the channel axis etc.
    - shift_step_size: Integer at least 1. The number of time frames between the time frames in the stack - 1. 
        For example, if set to 1 then zero time frames are skipped, if set to 2 then 1 is skipped etc.
    
    Outputs:
    - x: input to the neural network. Shape == [instance per batch, time frames per instance, channel count * shift_count]
    """

    # Shapes
    if (len(x.shape) == 2): x = x.unsqueeze(0)
    instances_per_batch, time_frames_per_instance, channel_count = x.size()

    # Pad it along the time axis
    extra_time_point_count = (shift_count - 1) * shift_step_size
    zeros = torch.zeros(size=[instances_per_batch, extra_time_point_count, channel_count])
    x = torch.cat([zeros,x], dim=1)
    
    # Stack
    xs = [None] * shift_count
    for p in range(len(xs)): xs[p] = x[:,p*shift_step_size:p*shift_step_size+time_frames_per_instance,:].unsqueeze(1)
    x = torch.cat(xs, dim=1)

    # Reshape 
    x = x.permute((0,2,1,3)).reshape((instances_per_batch, time_frames_per_instance, shift_count * channel_count)).squeeze()

    # Outputs
    return x

