import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import torch 
from scipy.io import wavfile

if __name__ == "__main__":
    root_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
    data_path = os.path.join(root_path, 'data', 'eeg_to_spectrogram')
    results_path = os.path.join(root_path, 'results', 'eeg_to_spectrogram')
    model_names = ["L", "D", "C", "R","A"]

    # Font
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams['font.family'] = "sans-serif"

    # Plot Correlations for spectrograms and split by model
    correlation_means = [None] * len(model_names)
    correlation_standard_errors = [None] * len(model_names)

    for m, model_name in enumerate(model_names):
        with open(os.path.join(results_path, model_name, 'qualities.txt'),'r') as text_file:
            lines = text_file.readlines() # Each line gives the subject name and mean correlation of original and reconstructed spectrogram
            lines = [float(line.split(', ')[1]) for line in lines] # Extracts the correlations
            correlation_means[m] = np.mean(lines)
            correlation_standard_errors[m] = np.std(lines) / (len(lines)**0.5)

    plt.figure()
    plt.title("Spectrogram Correlations per Model")
    model_abbreviations = [name.split(" ")[0] for name in model_names]
    plt.bar(x=model_abbreviations, height=correlation_means, yerr=correlation_standard_errors, capsize=7)
    plt.ylabel("Correlation"); plt.xlabel("Model")
    plt.ylim([0.45,0.75])
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'Correlations per Model.png'), dpi=600)
    plt.close()

    # Plot for one subject part of the original spectrogram, waveform and for each model the reconstruction
    subject = 'sub-06'
    spectrograms = [None] * (len(model_names) + 1)
    waveforms = [None] * (len(model_names) + 1)
    labels = np.load(os.path.join(data_path,f'{subject}_procWords.npy'))
    tick_locations = [0]
    tick_labels = [labels[0]]
    for l in range(1,len(labels)):
        if labels[l] != labels[l-1] and labels[l] != '':
            tick_locations.append(l)
            tick_labels.append(labels[l])
    tick_locations = np.array(tick_locations)
    tick_labels = np.array(tick_labels)

    spectrograms[0] = np.load(os.path.join(data_path,f'{subject}_spec.npy'))
    _, waveforms[0] = wavfile.read(os.path.join(results_path, model_names[0], f"{subject}_orig_synthesized.wav"))

    for m, model_name in enumerate(model_names):
        spectrograms[m+1] = np.load(os.path.join(results_path,model_name,f'{subject}_predicted_spec.npy'))
        _, waveforms[m+1] = wavfile.read(os.path.join(results_path,model_name,f'{subject}_predicted.wav'))
    
    titles = ["Original"] + model_names
    seconds = 10
    plt.figure()
    plt.subplot(len(spectrograms),2,1); plt.title("Spectrograms")
    for s in range(len(spectrograms)):
        plt.subplot(len(spectrograms),2,2*s+1)
        stop = (int)(seconds*22050/256)
        plt.imshow(np.flipud(spectrograms[s][:stop].T), aspect='auto')
        plt.yticks([])
        plt.ylabel(titles[s].split(' ')[0])
        if s < len(spectrograms) -1: plt.xticks([])
        else:  plt.xticks(tick_locations[tick_locations < stop], tick_labels[tick_locations < stop])
    plt.xlabel("Time")

    plt.subplot(len(spectrograms),2,2); plt.title("Waveforms")
    for s in range(len(spectrograms)):
        plt.subplot(len(spectrograms),2,2*s+2)
        plt.plot(waveforms[s][:(int)(seconds*22050)])
        plt.yticks([])
        if s < len(spectrograms) -1: plt.xticks([])
        else:  plt.xticks(tick_locations[tick_locations < stop]*256, tick_labels[tick_locations < stop])
    
    plt.xlabel("Time")
    plt.suptitle("Example Predictions per Model")
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'Example Predictions per Model ({subject}).png'), dpi=600)
    plt.close()

    # Plot the processing times split by model and slice size
    speed_means = {}
    speed_standard_errors = {}
    for m, model_name in enumerate(model_names):
        with open(os.path.join(results_path, model_name, 'speeds.txt'),'r') as text_file:
            lines = text_file.readlines() # Each line gives the subject name and speeds for several slize sizes
            # Now for each subject we have one row containing the speeds for these slice sizes
            tmp = np.array([[float(num) for num in line.split(', ')[1:]] for line in lines]) # Extracts the speeds
            speed_means[model_name] = np.mean(tmp, 0)
            speed_standard_errors[model_name] = np.std(tmp,0)/(tmp.shape[0]**0.5)
    
    speed_means = pd.DataFrame(speed_means)
    speed_standard_errors = pd.DataFrame(speed_standard_errors)
    
    men_means = [20, 34, 30, 35, 27]
    women_means = [25, 32, 34, 20, 25]

    x = np.arange(len(model_names))  # the label locations
    
    fig, ax = plt.subplots()
    slice_count = len(speed_means)
    width = 0.5/slice_count  # the width of the bars
    for s in range(slice_count):
        rects = ax.bar(x-(slice_count/2)*width + s*width, 
            1000*speed_means.iloc[s], width, 
            label=f"{np.round(1000*((s+1)*256/22050),2)} ms", yerr=1000*speed_standard_errors.iloc[s], capsize=3)
        

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Time (ms)')
    ax.set_title('Processing Times per Model')
    plt.xticks(x, model_abbreviations)
    plt.xlabel("Model")
    ax.legend()

    fig.tight_layout()
    plt.savefig(os.path.join(results_path, 'Processing Time per Model.png'), dpi=600)
    plt.close()
    

            