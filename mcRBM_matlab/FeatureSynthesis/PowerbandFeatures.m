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

% Frequency ranges for each band
f_all = [0.1 24];
f_delta = [0.5 4];
f_theta = [6 12];
f_beta = [12 23];

% Converting into log space
numfreqs = 100;
FFTfreqs = logspace(log10(f_all(1)),log10(f_all(2)),numfreqs);

Fs = % sampling rate
window = 2;
noverlap = 0;

[thFFTspec,thFFTfreqs,t_FFT] = spectrogram(data,window*Fs,noverlap*Fs,FFTfreqs,Fs);
thFFTspec = (abs(thFFTspec));
[zFFTspec,mu,sig] = zscore(log10(thFFTspec)');
thfreqs = find(thFFTfreqs>=f_theta(1) & thFFTfreqs<=f_theta(2));
thpower = sum((thFFTspec(thfreqs,:)),1);
allpower = sum((thFFTspec),1);

thratio = thpower./allpower;    %Narrowband Theta
    