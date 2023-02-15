import torch
from typing import Dict, List, Type
import pandas as pd
import os, random
import sys, os, numpy as np
sys.path.append('.')
from models import neural_networks as mnn
import librosa
from pydub import AudioSegment
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import epitran.vector
import time

def make_phoneme_vectors(data_path: str, language_label: str = 'uzb-Latn', max_vector_count: int = 64) -> torch.Tensor:
    """Creates a tensor of unique phoneme feature vectors. 
    It takes sentences from other.tsv and validated.tsv and computes their phoneme feature vectors using epitran.
    It casts words to lowercase and filters out ',', '.', '?', '!', ':', '-'

    Inputs:
    - data_path: the path to the folder that contains the other.tsv and validated.tsv files.
    - language_label: the label that is used to indicate the language to epitran. See epitran documentation for details.
    - max_vector_count: the maximum number of unique vectors that the data base should accumulate. The smaller this value, the faster this function terminates.
    
    Outputs:
    - phoneme_vectors: tensor of shape [time frame count, feature count].
    - phonemes: list of phoneme labels with length [time frame count]."""

    # Load sentences
    other = pd.read_csv(os.path.join(data_path,"other.tsv"), delimiter="\t")
    validated = pd.read_csv(os.path.join(data_path, "validated.tsv"), delimiter="\t")
    df = pd.concat([other, validated], axis=0, ignore_index=True)
    sentences = list(df.sentence)
    del df

    # Clean sentences
    word_set = set([])
    for sentence in sentences:
        words = sentence.split(" ")
        for word in words:
            word = word.lower()
            for punctuation in ['!','?','.',',','-',':']: word = word.replace(punctuation, '')
            if word != '': word_set.add(word)
    word_set = list(word_set)

    # Get unqiue phoneme vectors
    phoneme_vectors = []
    phonemes = []
    word_to_phoneme_sequence = epitran.vector.VectorsWithIPASpace(language_label, [language_label])
    for word in word_set:
        segs = word_to_phoneme_sequence.word_to_segs(word)
        for seg in segs:
            phoneme = seg[3]
            vector = torch.Tensor(seg[-1]).unsqueeze(0) # phonological_feature_vector
            exists = False
            for vector2 in phoneme_vectors:
                if torch.allclose(vector2, vector) : exists = True

            if not exists: 
                phoneme_vectors.append(vector) # Shape == [1, feature count]
                phonemes.append(phoneme)
        if max_vector_count <= len(phoneme_vectors): break
    phoneme_vectors = torch.cat(phoneme_vectors, dim=0)

    # Outputs
    return phoneme_vectors[:max_vector_count,:], phonemes[:max_vector_count]

def sample_batch(speaker_to_file_names: Dict[str, List[str]], spec_folder: str, instances_per_batch: int, from_mp3: bool = True, precision: Type = torch.float16) -> torch.Tensor:
    """Takes a set of arbitrary speakers from the pool of speakers and returns two distinct voice recordings for each of them.
    
    Inputs:
    - speaker_to_file_names: A dictionary that provides for each speaker identifier a list of paths to the speaker's recordings. 
        Assumes that each such list contains at least two paths.
    - recording_folder_path: The path to the folder that contains the recordings.
    - instances_per_batch: the number of instances that shall be in the batch. Since speakers are sampled arbitrarily a speaker may occur multiple times within the batch.
    - from_mp3: indicates whether the spectrogram should be generated direclty from the mp3 (takes more processing time) or loaded from preprocessed spectrogram files.
    - precision: the torch precision of the spectrogram.

    Outputs:
    - x_content, x_style: torch tensor with the mel spectrogram of the first and second recordings, respectively. Shape == [instances_per_batch, time point count, 80] where 80 is the number of mel features."""
    
    # Variables
    x_content = [None]*instances_per_batch; x_style = [None] * instances_per_batch
    alpha = 0 # This value will be used as constant for padding

    # Sample instances
    for i in range(instances_per_batch):
        if from_mp3:
            x_content[i], x_style[i] = sample_instance_from_mp3_file(speaker_to_mp3_paths=speaker_to_file_names, mp3_folder_path=spec_folder)
        else:
            x_content[i], x_style[i] = sample_instance_from_spec_file(speaker_to_spectrogram_paths=speaker_to_file_names, spectrogram_folder_path=spec_folder)
        
        # Cast to precision
        x_content[i] = x_content[i].to(dtype=precision)
        x_style[i] = x_style[i].to(dtype=precision)

        # Update alpha by the value that the spectrograms have for silence
        alpha += x_content[i][0,0] + x_style[i][0,0]

        # Flip x such that padding results in prepadding
        x_content[i] = torch.flipud(x_content[i])
        x_style[i] = torch.flipud(x_style[i])

    # Update alpha
    alpha = 0 if instances_per_batch <= 0 else alpha / (2*instances_per_batch)

    # Pad
    x_content = pad_sequence(x_content, padding_value=alpha) # This is the value that spectrograms have for silence
    x_style = pad_sequence(x_style, padding_value=alpha) 

    # Permute x_style to complete prepadding
    x_content = torch.flipud(x_content).permute(dims=[1,0,2])
    x_style = torch.flipud(x_style).permute(dims=[1,0,2])

    # Outputs
    return x_content, x_style
        
def sample_instance_from_spec_file(speaker_to_spectrogram_paths: Dict[str, List[str]], spectrogram_folder_path: str) -> torch.Tensor:
    """Takes an arbitrary speaker from the pool of speakers and returns two distinct spectrograms from that speaker.
    
    Inputs:
    - speaker_to_spectrogram_paths: A dictionary that provides for each speaker identifier a list of paths to the speaker's spectrograms. 
        Assumes that each such list contains at least two paths.
    - spectrogram_folder_path: The path to the folder that contains all spectrograms.

    Outputs:
    - x_content, x_style: torch tensor with the mel spectrogram of the first and second recordings, respectively. Shape == [time point count, 80] where 80 is the number of mel features."""
    
    # Sample speaker
    speaker = random.sample(list(speaker_to_spectrogram_paths.keys()), k=1)[0]

    # Sample 2 recording paths
    spectrogram_paths = random.sample(speaker_to_spectrogram_paths[speaker], k=2) # These are unique samples

    # Load the spectrogram
    x_content = torch.Tensor(pd.read_csv(os.path.join(spectrogram_folder_path, spectrogram_paths[0] + ".csv")).values)
    x_style = torch.Tensor(pd.read_csv(os.path.join(spectrogram_folder_path, spectrogram_paths[1] + ".csv")).values)

    # Outputs
    return x_content, x_style

def sample_instance_from_mp3_file(speaker_to_mp3_paths: Dict[str, List[str]], mp3_folder_path: str) -> torch.Tensor:
    """Takes an arbitrary speaker from the pool of speakers and returns two distinct spectrograms from that speaker.
    
    Inputs:
    - speaker_to_mp3_paths: A dictionary that provides for each speaker identifier a list of paths to the speaker's mp3 files. 
        Assumes that each such list contains at least two paths.
    - mp3_folder_path: The path to the folder that contains all mp3s.

    Outputs:
    - x_content, x_style: torch tensor with the mel spectrogram of the first and second recordings, respectively. Shape == [time point count, 80] where 80 is the number of mel features."""
    
    # Sample speaker
    speaker = random.sample(list(speaker_to_mp3_paths.keys()), k=1)[0]

    # Sample 2 recording paths
    recording_paths = random.sample(speaker_to_mp3_paths[speaker], k=2) # These are unique samples

    # Load the mp3
    recording_0 = AudioSegment.from_mp3(os.path.join(mp3_folder_path, recording_paths[0] + ".mp3"))
    recording_1 = AudioSegment.from_mp3(os.path.join(mp3_folder_path, recording_paths[1] + ".mp3"))
    
    # Convert to spectrogram
    x_content = mnn.VocGan.waveform_to_mel_spectrogram(waveform=recording_0.get_array_of_samples(), original_sampling_rate=recording_0.frame_rate).T
    x_style = mnn.VocGan.waveform_to_mel_spectrogram(waveform=recording_1.get_array_of_samples(), original_sampling_rate=recording_1.frame_rate).T
        
    # Outputs
    return x_content, x_style
   
def mp3_to_spec_files(input_folder:str, output_folder:str)-> None:
    """Converts all wav files to numpy files saving the spectrograms in the given folder.
    
    Inputs:
    - input_folder: The folder where the wav files are located.
    - output_folder: The folder where the numpy files shall be stored.
    
    Outputs:
    - None"""
    # Get all paths
    paths = os.listdir(input_folder)

    # Create output folder
    if not os.path.exists(output_folder): os.mkdir(output_folder)
    t = 1
    # Convert
    for path in paths:
        recording = AudioSegment.from_mp3(os.path.join(input_folder, path))
        x = mnn.VocGan.waveform_to_mel_spectrogram(waveform=recording.get_array_of_samples(), original_sampling_rate=recording.frame_rate).T
        pd.DataFrame(x.to(torch.float16).detach().numpy()).to_csv(os.path.join(output_folder, path.replace(".mp3", ".csv")), index=False)
        time.sleep(t)

def make_speaker_to_file_names(data_path: str)->None:
    """Creates a dictionary that uses as keys the speaker identifiers and as values the names to their files.
    It considers the records listed in other.tsv and validated.tsv.

    Inputs:
    - data_path: Path to the folder that contains the files other.tsv and validated.tsv.
    
    Outputs:
    - speaker_to_file_names: dictionary where each key is a speaker identifier and each value is a list of names for the files."""

    # Load speaker data
    other = pd.read_csv(os.path.join(data_path,"other.tsv"), delimiter="\t")
    validated = pd.read_csv(os.path.join(data_path, "validated.tsv"), delimiter="\t")
    df = pd.concat([other, validated], axis=0, ignore_index=True)
    
    # For each speaker get their recording paths
    speaker_to_file_names = {}
    speakers = list(df['client_id'].unique())
    for speaker in speakers:
        recording_names = list(df.loc[df['client_id'] == speaker]['path'])
        if 2 <= len(recording_names):
            speaker_to_file_names[speaker] = [name.replace(".mp3","") for name in recording_names]

    return speaker_to_file_names

if __name__ == "__main__":
    # Constants
    data_path = "data/speech_to_speech"
    
    # Convert mp3 to spectrograms (only done once for the entire dataset if you want to preprocess them beforehand (takes a lot of storage)
    #mp3_to_spec_files(input_folder=os.path.join(data_path,"clips"), output_folder=os.path.join(data_path, "clips_spec"))

    # Get the spectrogram names for each speaker
    speaker_to_file_names = make_speaker_to_file_names(data_path=data_path)
    
    # Sample a batch from the data
    x_content, x_style = sample_batch(speaker_to_file_names=speaker_to_file_names, spec_folder=os.path.join(data_path,"clips_spec"), instances_per_batch=4, from_mp3=False)
    
    # Plot the first entry of the batch
    plt.figure(); plt.suptitle("Example instance")
    plt.subplot(2,1,1); plt.imshow(x_content[0].to(torch.float32).T); plt.title("Content Spectrogram")
    plt.subplot(2,1,2); plt.imshow(x_style[0].to(torch.float32).T); plt.title("Style Spectrogram"); plt.show()

    # Create an alphabet of phoneme vectors
    x_alphabet, phonemes = make_phoneme_vectors(data_path=data_path, language_label='nld-Latn', max_vector_count=40)
    
    # Stretch it along the channel axis
    phoneme_vector_count = x_alphabet.size()[0]
    x_alphabet = x_alphabet.unsqueeze(1).repeat(repeats=[1,5,1]).permute([0,2,1]).reshape([phoneme_vector_count,24*5])
    
    # Concatenate with ones to get 128 features
    x_alphabet = torch.cat([x_alphabet, torch.ones(phoneme_vector_count, 8)], dim=-1)

    # Plot
    plt.figure(); plt.title('Alphabet')
    plt.imshow(x_alphabet.detach().numpy().T, aspect='auto')
    plt.xlabel('Phonemes'); plt.ylabel('Features'); 
    plt.xticks(list(range(phoneme_vector_count)), phonemes); plt.show()

    # Create neural network
    model = mnn.SpeechAutoEncoder(input_feature_count=x_content.shape[-1], x_alphabet=x_alphabet)

    # Fit
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = torch.nn.MSELoss()
    epoch_count = 2
    instances_per_batch = 16
    losses = [0.0] * epoch_count
    batch_count = len(os.listdir(os.path.join(data_path, "clips"))) // instances_per_batch
    model_progress_path = "parameters/voice_auto_encoder"
    e = 0

    # Set true if continuing training session, false if starting from scratch
    if True:
        checkpoint = torch.load(model_progress_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        e = checkpoint['epoch']
        losses = checkpoint['losses']
    
    for e in range(e, epoch_count):  # loop over the dataset multiple times

        for b in range(batch_count):
            # get the inputs
            x_content, x_style = sample_batch(speaker_to_file_names=speaker_to_file_names, spec_folder=os.path.join(data_path,"clips_spec"), instances_per_batch=16)
    
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_hat = model([x_content, x_style])
            loss = criterion(y_hat, x_content)
            loss.backward()
            optimizer.step()
            losses[e] += loss.item() / batch_count
            
            # print statistics
            if b % 10 == 9: print(f'epoch {e + 1}, batch {b + 1}, loss {loss.item()}')

        # Save progress
        torch.save({'epoch': e, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'losses': losses,}, model_progress_path + f"_epoch_{e}.torch")

    print('Finished Training')

    # Predict
    y_hat = model([x_content, x_style]).detach().numpy()

    # Plot example prediction
    plt.figure(); plt.suptitle("Example Prediction")
    plt.subplot(2,1,1); plt.title('Content Input'); plt.imshow(x_content[0].T); plt.ylabel('Mel Channels')
    plt.subplot(2,1,2); plt.title('Content Output'); plt.imshow(y_hat[0].T); plt.ylabel('Mel Channels')
    plt.xlabel("Time"); plt.show()

    k=2