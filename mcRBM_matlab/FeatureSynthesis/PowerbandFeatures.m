%% Synthesis of power band features for each band from the raw sleep dataset

clear; close all; clc
[DataPFC, TimeVectLFP, HeadingData] = load_open_ephys_data_faster('100_CH2_0.continuous');
[DataHPC, ~, ~] = load_open_ephys_data_faster('100_CH53_0.continuous');
% extracting the sampling frequency of the data
SamplingFreq = HeadingData.header.sampleRate;       % Sampling frequency of the data
% Downsample the data to different sampling rates for fast processing
TargetSampling1 = 600;                             % The goal sampling
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

for i=1:numEpochs
    lfpPFCEpoch = lfpPFCDown(1*i:epochSampLen*i);
    [FFTspec,f_FFT,t_FFT] = spectrogram(data,window*Fs,noverlap*Fs,FFTfreqs,Fs);
    FFTspec = (abs(FFTspec));
    allpower = sum((FFTspec),1);

    thfreqs = find(f_FFT>=f_theta(1) & f_FFT<=f_theta(2));
    thpower = sum((FFTspec(thfreqs,:)),1);
    

    thratio = thpower./allpower;    %Narrowband Theta
end


    