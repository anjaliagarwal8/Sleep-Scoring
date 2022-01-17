%% Synthesis of power band features for each band from the raw sleep dataset

clear; close all; clc
[DataHPC, TimeVectLFP, HeadingData] = load_open_ephys_data_faster('100_CH2_0.continuous');
[DataPFC, ~, ~] = load_open_ephys_data_faster('100_CH53_0.continuous');
% extracting the sampling frequency of the data
SamplingFreq = HeadingData.header.sampleRate;       % Sampling frequency of the data
% Downsample the data to different sampling rates for fast processing
TargetSampling1 = 1250;                             % The goal sampling rate
timesDownSamp1  = SamplingFreq / TargetSampling1;   % Number of times of downsample the data
lfpPFCDown = decimate(DataPFC,timesDownSamp1,'FIR');
lfpHPCDown = decimate(DataHPC,timesDownSamp1,'FIR');
timVect1 = linspace(0,numel(lfpPFCDown)/TargetSampling1,numel(lfpPFCDown));

% Creating epochs for assigning it to various substates later
epochTimVec = 1; %s
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
noverlap = window-1;

% parameteres for EMG-like signal
samplingFrequencyEMG = 5;
smoothWindowEMG = 10;
matfilename = 'EMGLikeSignalMat';

lfpFeatures = zeros(numEpochs,4);
for i=1:numEpochs
    lfpPFCEpoch = lfpPFCDown((i-1)*epochSampLen+1:i*epochSampLen);
    lfpHPCEpoch = lfpHPCDown((i-1)*epochSampLen+1:i*epochSampLen);
    
    % 
    [FFTspec,f_FFT,~] = spectrogram(lfpPFCEpoch,window*Fs,noverlap*Fs,FFTfreqs,Fs);
    FFTspec = (abs(FFTspec));
    
    % delta power
    delfreqs = f_FFT>=f_delta(1) & f_FFT<=f_delta(2);
    delpower = sum((FFTspec(delfreqs,:)),1);

%     % theta power
%     thfreqs = f_FFT>=f_theta(1) & f_FFT<=f_theta(2);
%     thpower = sum((FFTspec(thfreqs,:)),1);
    
    % beta power
    betafreqs = f_FFT>=f_beta(1) & f_FFT<=f_beta(2);
    betapower = sum((FFTspec(betafreqs,:)),1);
   
%     % Calculating the ratios....
%     lfpFeatures(i,1) = delpower/thpower;
%     lfpFeatures(i,2) = delpower/betapower;
%     lfpFeatures(i,3) = thpower/betapower;

    % 
    [FFTspec,f_FFT,t_FFT] = spectrogram(lfpHPCEpoch,window*Fs,noverlap*Fs,FFTfreqs,Fs);
    FFTspec = (abs(FFTspec));

%     % delta power
%     delfreqs = find(f_FFT>=f_delta(1) & f_FFT<=f_delta(2));
%     delpower = sum((FFTspec(delfreqs,:)),1);

    % theta power
    thfreqs = find(f_FFT>=f_theta(1) & f_FFT<=f_theta(2));
    thpower = sum((FFTspec(thfreqs,:)),1);
    
%     % beta power
%     betafreqs = find(f_FFT>=f_beta(1) & f_FFT<=f_beta(2));
%     betapower = sum((FFTspec(betafreqs,:)),1);
    
    % Calculating the ratios....
    lfpFeatures(i,1) = delpower/thpower;
    lfpFeatures(i,2) = delpower/betapower;
    lfpFeatures(i,3) = thpower/betapower;

    % Calculating EMG-like signal
    EMGFromLFP = compute_emg_buzsakiMethod(samplingFrequencyEMG, Fs, lfpPFCEpoch, lfpHPCEpoch, smoothWindowEMG,matfilename);
    % Integrating the EMG signal for each epoch
    EMGSig = trapz(EMGFromLFP.smoothed);

    lfpFeatures(i,4) = EMGSig;
end

save Features.mat lfpFeatures

% Preprocessing the features by taking log and zeroing the mean of each
% feature
[PreprocessedFeatures,~,~] = zscore(log(lfpFeatures));

save PreprocessedFeatures.mat PreprocessedFeatures
    
% Manually scored states
States = load('2019-06-06_13-26-20_Post-Trial3-states.mat');
downsampledStates = downsample(States.states,8);
%downsampledStates(1,110) = 1; 

save states.mat downsampledStates

state_1 = downsampledStates == 1; %Wake
state_3 = downsampledStates == 3; %NREM
state_4 = downsampledStates == 4; %NREM-REM
state_5 = downsampledStates == 5; %REM

Features_1 = PreprocessedFeatures(state_1,:);
Features_3 = PreprocessedFeatures(state_3,:);
Features_4 = PreprocessedFeatures(state_4,:);
Features_5 = PreprocessedFeatures(state_5,:);

[status, msg, msgID] = mkdir('FeatureHistograms');
cd FeatureHistograms

features = ["Ch2__del__theta","Ch2__del__beta","Ch2__theta__beta","Ch53__del__theta","Ch53__del__beta","Ch53__theta__beta","EMG-like"];
for i=1:7
    subplot(2,2,1)
    histogram(Features_1(:,i),'BinMethod','fd')
    title('WAKE')
    subplot(2,2,2)
    histogram(Features_3(:,i),'BinMethod','fd')
    title('NREM')
    subplot(2,2,3)
    histogram(Features_4(:,i),'BinMethod','fd')
    title('NREM-REM')
    subplot(2,2,4)
    histogram(Features_5(:,i),'BinMethod','fd')
    title('REM')
    sgtitle(features(i))
    saveas(gcf,features(i)+".png")
end
