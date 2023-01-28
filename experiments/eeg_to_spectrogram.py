import os, sys, time
sys.path.append('.')
import models.neural_networks as mnn
import models.fitter as mft
import models.utilities as mut
import numpy as np, torch
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from typing import List

def compute_speeds(neural_network, feature_count, slice_size_sount=5, repeats = 100):
    speeds = [0] * slice_size_sount
    for i in range(len(speeds)):
        x_i = torch.rand(size=(1,i+1,feature_count))
        
        for j in range(repeats):
            tick = time.time() 
            _ = neural_network(x_i)
            tock = time.time()
            speeds[i] += (tock-tick) / repeats

    return speeds

def count_parameters(neural_network):
    return sum(p.numel() for p in neural_network.parameters() if p.requires_grad)

def plot_x_target_and_output(x: torch.Tensor, target: torch.Tensor, output: torch.Tensor, labels: List[str], pause_string: str, path: str, model_name: str) -> None:
    """Plots x, the target and output.
    Inputs:
    - x: Input EEG time series. Shape == [time frame count, eeg channel count].
    - target: Desired spectrogram. Shape == [time frame count, mel channel count].
    - output: Obtained spectrogram. Shape == [time frame count, mel channel count].
    - labels: The labels that indicate for each time frame of x, target and output which label was present at that time. Length == time frame count.
    - pause_string: The string used to indicate pauses.
    - path: Path to the folder where the figure should be stored.
    - model_name: The name of the model used for the figure title.

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
    plt.suptitle("Sample of Data Passed Through " + model_name)

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
    plt.tight_layout()
    plt.savefig(os.path.join(path, "Sample Data.png"), dpi=600)
    plt.close(fig)

def plot_loss_trajectory(train_losses: List[float], validation_losses: List[float], path: str, 
                        loss_name: str, model_name: str, logarithmic: bool = True) -> None:
    """Plots the losses of train and validation time courses per epoch.
    
    Assumptions:
    - train and validation losses are assumed to have the same number of elements and that their indices are synchronized.

    Inputs:
    - train_losses: The losses of the model during training.
    - validation_losses: The losses of the model during validation.
    - path: Path to the folder where the figure should be stored.
    - loss_name: Name of the loss function.
    - model_name: Name of the model used for the title.
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
    plt.title("Learning curve for " + model_name)
    plt.xlabel("Epoch"); plt.ylabel(loss_name)

    # Save
    if not os.path.exists(path): os.makedirs(path)
    plt.tight_layout()
    plt.savefig(os.path.join(path, "Learning Curve.png"), dpi=600)
    plt.close(fig=fig)

if __name__=="__main__":
    # Adapted from https://github.com/neuralinterfacinglab/SingleWordProductionDutch
    root_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
    data_path = os.path.join(root_path, 'data', 'eeg_to_spectrogram')
    result_path = os.path.join(root_path, 'results', 'eeg_to_spectrogram')
    participant_ids = ['sub-%02d'%i for i in range(1,2)]

    # Cross validation
    fold_count = 2
    kf = KFold(fold_count,shuffle=False)
    
    # VocGan
    voc_gan = mnn.VocGan.load(is_streamable=False)

    # Initialize empty matrices for correlation results, randomized contols and amount of explained variance
    all_results = np.zeros((len(participant_ids),fold_count,80))
    
    for p, participant_id in enumerate(participant_ids):
        # Load the data
        spectrogram = torch.Tensor(np.load(os.path.join(data_path,f'{participant_id}_spec.npy')))
        data = torch.Tensor(np.load(os.path.join(data_path,f'{participant_id}_feat.npy')))
        labels = np.load(os.path.join(data_path,f'{participant_id}_procWords.npy'))
        
        # Since shapes are not exactly the same due to jittering we discard trailing time frames    
        min_time_frame_count = np.min([spectrogram.size()[0], data.size()[0], len(labels)])
        spectrogram = spectrogram[:min_time_frame_count,:]
        data = data[:min_time_frame_count,:]
        labels = labels[:min_time_frame_count]
 
        #Initialize an empty spectrogram to save the reconstruction to
        rec_spec = np.zeros(spectrogram.shape)
        #Save the correlation coefficients for each fold
        rs = np.zeros((fold_count,spectrogram.shape[1]))
        for k,(train, test) in enumerate(kf.split(data)):
            # Normalize input
            data=(data-torch.mean(data,axis=1).unsqueeze(1))/torch.std(data,axis=1).unsqueeze(1)
            
            # Extract slices
            train_spectrogram, test_spectrogram = spectrogram[train,:], spectrogram[test,:]
            train_data, test_data = data[train,:], data[test,:]
            train_labels, test_labels = list(labels[train]), list(labels[test])

            # Construct neural networks
            eeg_channel_count = data.shape[1]; mel_channel_count = spectrogram.shape[1]
            step_count = 8; step_size = 8
            #stationary_neural_network = mnn.MemoryDense(input_feature_count=eeg_channel_count, output_feature_count=mel_channel_count, step_count=step_count, step_size=step_size, layer_count=1, name="L")
            #stationary_neural_network = mnn.MemoryDense(input_feature_count=eeg_channel_count, output_feature_count=mel_channel_count, step_count=step_count, step_size=step_size, layer_count=2, name="D")
            #stationary_neural_network = mnn.Convolutional(input_feature_count=eeg_channel_count, output_feature_count=mel_channel_count, name='C')
            stationary_neural_network = mnn.Recurrent(input_feature_count=eeg_channel_count, output_feature_count=mel_channel_count, name="R")
            #stationary_neural_network = nns.AttentionNeuralNetwork(query_feature_count=eeg_channel_count, hidden_feature_count=eeg_channel_count, x_key=train_data[:1024,:], x_value=train_spectrogram[:1024,:], labels=train_labels[:1024], pause_string='', shift_count=shift_count, shift_step_size=shift_step_size)
       
            # Train
            loss_function = torch.nn.MSELoss()
            L2_weight = 1e-2
            
            optimizer = torch.optim.Adam(stationary_neural_network.parameters(), lr=1e-2, weight_decay=L2_weight)
            fitter = mft.Fitter(is_streamable=False)
            train_data, train_spectrogram = mut.reshape_by_label(x=train_data, labels=train_labels, pause_string='', y=train_spectrogram)
            train_losses, validation_losses = fitter.fit(stationary_neural_network=stationary_neural_network, x=train_data, y=train_spectrogram, loss_function=loss_function, optimizer=optimizer, epoch_count=50)
            time.sleep(2) # To prevent CPU from overheating
            prediction = stationary_neural_network.predict(x=test_data.unsqueeze(0)).squeeze()

            # Plots
            model_name = stationary_neural_network.name
            plot_loss_trajectory(train_losses=train_losses, validation_losses=validation_losses, path=os.path.join(result_path, model_name, participant_id), loss_name=f"mean squared error + {L2_weight}*L2 norm", model_name=model_name,logarithmic=False)
            plot_x_target_and_output(x=test_data[:1000,:], target=test_spectrogram[:1000,:], output=prediction[:1000,:], labels=test_labels[:1000], pause_string='', path=os.path.join(result_path, model_name, participant_id), model_name=model_name)

            #Predict the reconstructed spectrogram for the test data
            rec_spec[test, :] = prediction.detach().numpy()

            #Evaluate reconstruction of this fold
            for specBin in range(spectrogram.shape[1]):
                if np.any(np.isnan(rec_spec)):
                    print('%s has %d broken samples in reconstruction' % (participant_id, np.sum(np.isnan(rec_spec))))
                r, _ = pearsonr(spectrogram[test, specBin], rec_spec[test, specBin])
                rs[k,specBin] = r

        # Save quality
        quality_mean = np.mean(rs) # mean over this participant's folds and spectral bins
        quality_std = np.std(rs) # standard deviation
        qualities_path = os.path.join(result_path, model_name, 'qualities.txt')
        new_lines = []
        if os.path.exists(qualities_path):
            with open(qualities_path, 'r') as text_file:
                lines = text_file.readlines()
                
                for l, line in enumerate(lines):
                    if participant_id not in line: 
                        if '\n' not in line: line += '\n'
                        new_lines.append(line)
        new_lines.append(f"{participant_id}, {quality_mean}, {quality_std}")
            
        with open(qualities_path, 'w') as text_file:
            text_file.writelines(new_lines)

        # Save speed
        speeds = compute_speeds(neural_network=stationary_neural_network, feature_count=eeg_channel_count)
        speeds = [str(speed) for speed in speeds]
        speeds_path = os.path.join(result_path, model_name, 'speeds.txt')
        new_lines = []
        if os.path.exists(speeds_path):
            with open(speeds_path, mode='r') as text_file:
                lines = text_file.readlines()
                
                for l, line in enumerate(lines):
                    if participant_id not in line: 
                        if '\n' not in line: line += '\n'
                        new_lines.append(line)
        new_lines.append(f"{participant_id}, " + ', '.join(speeds))
            
        with open(speeds_path, mode='w') as text_file:
            text_file.writelines(new_lines)
            

        #Show evaluation result
        print('%s has mean correlation of %f' % (participant_id, np.mean(rs)))
        all_results[p,:,:]=rs

        #Save reconstructed spectrogram
        os.makedirs(os.path.join(result_path), exist_ok=True)
        np.save(os.path.join(result_path,model_name, f'{participant_id}_predicted_spec.npy'), rec_spec)
        
        #Synthesize waveform from spectrogram using Vocgan
        reconstructedWav, _ = voc_gan.mel_spectrogram_to_waveform(mel_spectrogram=torch.Tensor(rec_spec[:1024,:].T)) # only synthesize the first few secodns to save some time
        mnn.VocGan.save(waveform= reconstructedWav, file_path=os.path.join(result_path,model_name,f'{participant_id}_predicted.wav'))

        #For comparison synthesize the original spectrogram with Vocgan
        origWav, _ = voc_gan.mel_spectrogram_to_waveform(mel_spectrogram=torch.Tensor(spectrogram[:1024,:].T)) # only synthesize the first few secodns to save some time
        mnn.VocGan.save(waveform=origWav, file_path=os.path.join(result_path,model_name,f'{participant_id}_orig_synthesized.wav'))

    #Save results in numpy arrays          
    np.save(os.path.join(result_path,model_name, 'linearResults.npy'),all_results)
    