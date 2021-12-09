%% Synthesis of power band features for each band from the raw sleep dataset

clear; close all; clc
[DataPFC, TimeVectLFP, HeadingData] = load_open_ephys_data_faster('100_CH2_0.continuous');
[DataHPC, ~, ~] = load_open_ephys_data_faster('100_CH53_0.continuous');
% extracting the sampling frequency of the data
SamplingFreq = HeadingData.header.sampleRate;       % Sampling frequency of the data
% Downsample the data to different sampling rates for fast processing
TargetSampling1 = 1250;                             % The goal sampling rate
timesDownSamp1  = SamplingFreq / TargetSampling1;   % Number of times of downsample the data
lfpPFCDown = decimate(DataPFC,timesDownSamp1,'FIR');
lfpHPCDown = decimate(DataHPC,timesDownSamp1,'FIR');
timVect1 = linspace(0,numel(lfpPFCDown)/TargetSampling1,numel(lfpPFCDown));

% Creating epochs for assigning it to various substates later
epochTimVec = 3.1; %s
epochSampLen = epochTimVec*TargetSampling1;
numEpochs = floor(length(lfpHPCDown)/epochSampLen);

% Frequency ranges for each band
f_all = [0.1 24];
f_delta = [0.5 4];
f_theta = [6 12];
f_beta = [12 23];

% Log-Spaced frequency values for getting the spectrogram
numfreqs = 100;
FFTfreqs = logspace(log10(f_all(1)),log10(f_all(2)),numfreqs);

% Parameters for getting the spectrogram
Fs = TargetSampling1; % sampling rate
window = 2; 
noverlap = 0;

% parameteres for EMG-like signal
samplingFrequencyEMG = 5;
smoothWindowEMG = 10;
matfilename = 'EMGLikeSignalMat';

lfpFeatures = zeros(numEpochs,7);
for i=1:numEpochs
    lfpPFCEpoch = lfpPFCDown((i-1)*epochSampLen+1:i*epochSampLen);
    lfpHPCEpoch = lfpHPCDown((i-1)*epochSampLen+1:i*epochSampLen);
    
    % 
    [FFTspec,f_FFT,~] = spectrogram(lfpPFCEpoch,window*Fs,noverlap*Fs,FFTfreqs,Fs);
    FFTspec = (abs(FFTspec));
    
    % delta power
    delfreqs = f_FFT>=f_delta(1) & f_FFT<=f_delta(2);
    delpower = sum((FFTspec(delfreqs,:)),1);

    % theta power
    thfreqs = f_FFT>=f_theta(1) & f_FFT<=f_theta(2);
    thpower = sum((FFTspec(thfreqs,:)),1);
    
    % beta power
    betafreqs = f_FFT>=f_beta(1) & f_FFT<=f_beta(2);
    betapower = sum((FFTspec(betafreqs,:)),1);
   
    % Calculating the ratios....
    lfpFeatures(i,1) = delpower/thpower;
    lfpFeatures(i,2) = delpower/betapower;
    lfpFeatures(i,3) = thpower/betapower;

    % 
    [FFTspec,f_FFT,t_FFT] = spectrogram(lfpHPCEpoch,window*Fs,noverlap*Fs,FFTfreqs,Fs);
    FFTspec = (abs(FFTspec));

    % delta power
    delfreqs = find(f_FFT>=f_delta(1) & f_FFT<=f_delta(2));
    delpower = sum((FFTspec(delfreqs,:)),1);

    % theta power
    thfreqs = find(f_FFT>=f_theta(1) & f_FFT<=f_theta(2));
    thpower = sum((FFTspec(thfreqs,:)),1);
    
    % beta power
    betafreqs = find(f_FFT>=f_beta(1) & f_FFT<=f_beta(2));
    betapower = sum((FFTspec(betafreqs,:)),1);
    
    % Calculating the ratios....
    lfpFeatures(i,4) = delpower/thpower;
    lfpFeatures(i,5) = delpower/betapower;
    lfpFeatures(i,6) = thpower/betapower;

    % Calculating EMG-like signal
    EMGFromLFP = compute_emg_buzsakiMethod(samplingFrequencyEMG, Fs, lfpPFCEpoch, lfpHPCEpoch, smoothWindowEMG,matfilename);
    % Integrating the EMG signal for each epoch
    EMGSig = trapz(EMGFromLFP.smoothed);

    lfpFeatures(i,7) = EMGSig;
end

save Features.mat lfpFeatures

% Preprocessing the features by taking log and zeroing the mean of each
% feature
[PreprocessedFeatures,~,~] = zscore(log(lfpFeatures));

save PreprocessedFeatures.mat PreprocessedFeatures
    
% Manually scored states
States = load('2019-06-06_13-26-20_Post-Trial3-states.mat');
downsampledStates = downsample(States.states,25);
downsampledStates(1,110) = 1; 

state_0 = downsampledStates == 0;
state_1 = downsampledStates == 1;
state_2 = downsampledStates == 2;
state_3 = downsampledStates == 3;
state_4 = downsampledStates == 4;
state_5 = downsampledStates == 5;

Features_0 = PreprocessedFeatures(state_0,:);
Features_1 = PreprocessedFeatures(state_1,:);
Features_2 = PreprocessedFeatures(state_2,:);
Features_3 = PreprocessedFeatures(state_3,:);
Features_4 = PreprocessedFeatures(state_4,:);
Features_5 = PreprocessedFeatures(state_5,:);

[status, msg, msgID] = mkdir('FeatureHistograms');
cd FeatureHistograms

histogram(Features_1(:,1))
hold on
histogram(Features_3(:,1))
hold on
histogram(Features_4(:,1))
hold on
histogram(Features_5(:,1))
legend('WAKE','NREM','NREM-REM','REM')
title('Ch2__del__theta')
saveas(gcf,'Ch2_del_theta.png')

histogram(Features_1(:,2))
hold on
histogram(Features_3(:,2))
hold on
histogram(Features_4(:,2))
hold on
histogram(Features_5(:,2))
legend('WAKE','NREM','NREM-REM','REM')
title('Ch2__del__beta')
saveas(gcf,'Ch2_del_beta.png')

histogram(Features_1(:,3))
hold on
histogram(Features_3(:,3))
hold on
histogram(Features_4(:,3))
hold on
histogram(Features_5(:,3))
legend('WAKE','NREM','NREM-REM','REM')
title('Ch2__theta__beta')
saveas(gcf,'Ch2_theta_beta.png')

histogram(Features_1(:,4))
hold on
histogram(Features_3(:,4))
hold on
histogram(Features_4(:,4))
hold on
histogram(Features_5(:,4))
legend('WAKE','NREM','NREM-REM','REM')
title('Ch53__del__theta')
saveas(gcf,'Ch53_del_theta.png')

histogram(Features_1(:,5))
hold on
histogram(Features_3(:,5))
hold on
histogram(Features_4(:,5))
hold on
histogram(Features_5(:,5))
legend('WAKE','NREM','NREM-REM','REM')
title('Ch53__del__beta')
saveas(gcf,'Ch53_del_beta.png')

histogram(Features_1(:,6))
hold on
histogram(Features_3(:,6))
hold on
histogram(Features_4(:,6))
hold on
histogram(Features_5(:,6))
legend('WAKE','NREM','NREM-REM','REM')
title('Ch53__theta__beta')
saveas(gcf,'Ch53_theta_beta.png')

histogram(Features_1(:,7))
hold on
histogram(Features_3(:,7))
hold on
histogram(Features_4(:,7))
hold on
histogram(Features_5(:,7))
legend('WAKE','NREM','NREM-REM','REM')
title('EMG-like')
saveas(gcf,'EMG-like.png')
