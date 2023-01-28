import torch, numpy as np
import sys, os
sys.path.append('.')
from models import neural_networks as mnn
import scipy as scipy
from scipy.io import wavfile

if __name__ == "__main__":
    # File configuration
    root_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
    file_name = 'Dutch example.wav'
    data_path = os.path.join(root_path, 'data', 'spectrogram_to_waveform',file_name)
    results_path = os.path.join(root_path, 'results', 'spectrogram_to_waveform')
    results_path_stationary = os.path.join(results_path, 'stationary', file_name)
    results_path_streamable = os.path.join(results_path, 'streamable', file_name)

    # Load waveform
    samplerate, waveform = wavfile.read(data_path)
    print(f"The audio is {waveform.shape[-1]/samplerate} seconds long.")

    # Convert to spectrogram
    # It is also possible to use your own mel spectrogram. Be sure it adheres to mnn.VocGan.TIMING_CONVENTIONS
    mel_spectrogram = mnn.VocGan.waveform_to_mel_spectrogram(waveform=waveform, original_sampling_rate=samplerate)
    
    # stationary Demo (stationary means not streamable)
    # Load model
    stationary_model = mnn.VocGan.load(is_streamable=False)

    # Convert
    waveform_stationary, stationary_processing_time = stationary_model.mel_spectrogram_to_waveform(mel_spectrogram=mel_spectrogram)
    print(f"Converting the spectrogram to waveform took {stationary_processing_time} seconds in stationary mode.")

    # Save
    mnn.VocGan.save(waveform=waveform_stationary, file_path=results_path_stationary)

    # streamable Demo
    # Load model
    streamable_model = mnn.VocGan.load(is_streamable=True)

    # Setup a generator for the spectrogram slices
    time_frames_per_slice, actual_seconds_per_slice, slice_count = mnn.VocGan.slice_duration_to_frame_count(spectrogram_time_frame_count=mel_spectrogram.shape[-1], target_seconds_per_slice=0.035)
    print(f"Each spectrogram slice contains {time_frames_per_slice} frames.")
    print(f"Each spectrogram slice provides {actual_seconds_per_slice} seconds of new audio.") # Note, the output slices will have duration of frames_per_slice * mnn.VocGan.TIMING_CONVENTIONS['Seconds Per Spectrogram Hop'] which is a bit shorter. The surplus is saved in the state of the vocgan during streamable
    generator = streamable_model.slice_generator(mel_spectrogram=mel_spectrogram, time_frames_per_slice=time_frames_per_slice)
    
    # Containers
    waveform_streamable_slices = [None] * slice_count
    slice_processing_times = [None] * slice_count
    
    # Stream
    for i in range(slice_count):
        x_i, is_final_slice = next(generator)
        waveform_streamable_slices[i], slice_processing_times[i] = streamable_model.mel_spectrogram_to_waveform(mel_spectrogram=x_i, is_final_slice=is_final_slice)
    print(f"Converting the spectrogram to waveform took {np.sum(slice_processing_times)} seconds in streamable mode.")
    
    # Save
    waveform_streamable = torch.cat(waveform_streamable_slices, axis=-1)
    mnn.VocGan.save(waveform=waveform_streamable, file_path=results_path_streamable)

    # Plot
    mnn.VocGan.plot(mel_spectrogram=mel_spectrogram, waveform_stationary=waveform_stationary, waveform_streamable_slices=waveform_streamable_slices, slice_processing_times=slice_processing_times)
