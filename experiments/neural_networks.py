import os, sys, time
sys.path.append('.')
import models.neural_networks as mnn
import numpy as np, torch
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def compute_speeds(neural_network, feature_count, slice_size_sount=5, repeats = 100):
    speeds = [0] * slice_size_sount
    for i in range(len(speeds)):
        x_i = torch.rand(size=(1,i+1,feature_count))
        if neural_network.model_name in ['Linear Regression', "Dense Neural Network"]:
            x_i = neural_network.__stack_x__(x=x_i, shift_count=neural_network.shift_count, shift_step_size=neural_network.shift_step_size)

        
        for j in range(repeats):
            tick = time.time() 
            _ = neural_network.model(x_i)
            tock = time.time()
            speeds[i] += (tock-tick) / repeats

    return speeds

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__=="__main__":
    spec2wav = svg.StreamingVocGan(is_streaming=False)
    feat_path = r'/Users/timdick/Documents/Master_Internship_old/SingleWordProductionDutch/SingleWordProductionDutch-iBIDS/features'
    result_path = r'SingleWordProductionDutch/SingleWordProductionDutch-iBIDS/results'
    pts = ['sub-%02d'%i for i in range(1,11)]

    winLength = 1024/22050
    frameshift = 256/22050
    audiosr = 22050

    nfolds = 2
    kf = KFold(nfolds,shuffle=False)
    
    #Initialize empty matrices for correlation results, randomized contols and amount of explained variance
    allRes = np.zeros((len(pts),nfolds,80))
    
    for pNr, pt in enumerate(pts):
        # Load the data
        spectrogram = torch.Tensor(np.load(os.path.join(feat_path,f'{pt}_spec.npy')))
        data = torch.Tensor(np.load(os.path.join(feat_path,f'{pt}_feat.npy')))
        labels = np.load(os.path.join(feat_path,f'{pt}_procWords.npy'))
        featName = np.load(os.path.join(feat_path,f'{pt}_feat_names.npy'))
        
        # Since shapes are not exactly the same due to jittering we discard trailing time frames    
        min_time_frame_count = np.min([spectrogram.size()[0], data.size()[0], len(labels)])
        spectrogram = spectrogram[:min_time_frame_count,:]
        data = data[:min_time_frame_count,:]
        labels = labels[:min_time_frame_count]
 
        #Initialize an empty spectrogram to save the reconstruction to
        rec_spec = np.zeros(spectrogram.shape)
        #Save the correlation coefficients for each fold
        rs = np.zeros((nfolds,spectrogram.shape[1]))
        for k,(train, test) in enumerate(kf.split(data)):
            # Extract slices
            train_spectrogram, test_spectrogram = spectrogram[train,:], spectrogram[test,:]
            train_data, test_data = data[train,:], data[test,:]
            train_labels, test_labels = list(labels[train]), list(labels[test])

            #Z-Normalize with mean and std 
            #train_data=(train_data-torch.mean(train_data,axis=1).unsqueeze(1))/torch.std(train_data,axis=1).unsqueeze(1)
            #test_data=(test_data-torch.mean(test_data,axis=1).unsqueeze(1))/torch.std(test_data,axis=1).unsqueeze(1)
            mu=torch.mean(train_data,dim=0)
            std=torch.std(train_data,dim=0)
            train_data=(train_data-mu)/std
            test_data=(test_data-mu)/std

            # Construct neural networks
            eeg_channel_count = data.shape[1]; mel_channel_count = spectrogram.shape[1]
            shift_count = 8; shift_step_size = 8
            est = mnn.LinearRegression(input_feature_count=eeg_channel_count, output_feature_count=mel_channel_count, shift_count=shift_count, shift_step_size=shift_step_size)
            #est = nns.DenseNeuralNetwork(input_feature_count=eeg_channel_count, output_feature_count=mel_channel_count, shift_count=shift_count, shift_step_size=shift_step_size)
            #est = nns.ConvolutionalNeuralNetwork(input_feature_count=eeg_channel_count, output_feature_count=mel_channel_count, shift_count=8, shift_step_size=8)
            #est = nns.RecurrentNeuralNetwork(input_feature_count=eeg_channel_count, output_feature_count=mel_channel_count)
            #est = nns.AttentionNeuralNetwork(query_feature_count=eeg_channel_count, hidden_feature_count=eeg_channel_count, x_key=train_data[:1024,:], x_value=train_spectrogram[:1024,:], labels=train_labels[:1024], pause_string='', shift_count=shift_count, shift_step_size=shift_step_size)
       
            # Train
            loss_function = torch.nn.MSELoss()
            L2_weight = 1e-2
            
            optimizer = torch.optim.Adam(est.model.parameters(), lr=1e-2, weight_decay=L2_weight)
            train_losses, validation_losses = est.fit(x=train_data, y=train_spectrogram, labels=train_labels, pause_string='', loss_function=loss_function, optimizer=optimizer, epoch_count=50)
            time.sleep(20) # To prevent CPU from overheating
            prediction = est.predict(x=test_data, labels=test_labels, pause_string='')

            # Plots
            est.plot_loss_trajectory(train_losses=train_losses, validation_losses=validation_losses, path=os.path.join(result_path, est.model_name, pt), loss_name=f"mean squared error + {L2_weight}*L2 norm", logarithmic=False)
            est.plot_x_target_and_output(x=test_data[:1000,:], target=test_spectrogram[:1000,:], output=prediction[:1000,:], labels=test_labels[:1000], pause_string='', path=os.path.join(result_path, est.model_name, pt))

            #Predict the reconstructed spectrogram for the test data
            rec_spec[test, :] = prediction.detach().numpy()

            #Evaluate reconstruction of this fold
            for specBin in range(spectrogram.shape[1]):
                if np.any(np.isnan(rec_spec)):
                    print('%s has %d broken samples in reconstruction' % (pt, np.sum(np.isnan(rec_spec))))
                r, p = pearsonr(spectrogram[test, specBin], rec_spec[test, specBin])
                rs[k,specBin] = r

        # Save quality
        quality_mean = np.mean(rs) # mean over this participant's folds and spectral bins
        quality_std = np.std(rs) # standard deviation
        qualities_path = os.path.join(result_path, est.model_name, 'qualities.txt')
        new_lines = []
        if os.path.exists(qualities_path):
            with open(qualities_path, 'r') as text_file:
                lines = text_file.readlines()
                
                for l, line in enumerate(lines):
                    if pt not in line: 
                        if '\n' not in line: line += '\n'
                        new_lines.append(line)
        new_lines.append(f"{pt}, {quality_mean}, {quality_std}")
            
        with open(qualities_path, 'w') as text_file:
            text_file.writelines(new_lines)

        # Save speed
        speeds = compute_speeds(neural_network=est, feature_count=eeg_channel_count)
        speeds = [str(speed) for speed in speeds]
        speeds_path = os.path.join(result_path, est.model_name, 'speeds.txt')
        new_lines = []
        if os.path.exists(speeds_path):
            with open(speeds_path, mode='r') as text_file:
                lines = text_file.readlines()
                
                for l, line in enumerate(lines):
                    if pt not in line: 
                        if '\n' not in line: line += '\n'
                        new_lines.append(line)
        new_lines.append(f"{pt}, " + ', '.join(speeds))
            
        with open(speeds_path, mode='w') as text_file:
            text_file.writelines(new_lines)
            

        #Show evaluation result
        print('%s has mean correlation of %f' % (pt, np.mean(rs)))
        allRes[pNr,:,:]=rs

        #Save reconstructed spectrogram
        os.makedirs(os.path.join(result_path), exist_ok=True)
        np.save(os.path.join(result_path,est.model_name, f'{pt}_predicted_spec.npy'), rec_spec)
        
        
        #Synthesize waveform from spectrogram using Vocgan
        reconstructedWav, _ = spec2wav.mel_spectrogram_to_waveform(mel_spectrogram=torch.Tensor(rec_spec[:1024,:].T)) # only synthesize the first few secodns to save some time
        svg.StreamingVocGan.save(waveform= reconstructedWav, file_path=os.path.join(result_path,est.model_name,f'{pt}_predicted.wav'))

        #For comparison synthesize the original spectrogram with Vocgan
        origWav, _ = spec2wav.mel_spectrogram_to_waveform(mel_spectrogram=torch.Tensor(spectrogram[:1024,:].T)) # only synthesize the first few secodns to save some time
        svg.StreamingVocGan.save(waveform=origWav, file_path=os.path.join(result_path,est.model_name,f'{pt}_orig_synthesized.wav'))

    #Save results in numpy arrays          
    np.save(os.path.join(result_path,est.model_name, 'linearResults.npy'),allRes)
    