%% Synthesis of power band features for each band from the raw sleep dataset

clear; close all; clc
[DataHPC, TimeVectLFP, HeadingData] = load_open_ephys_data_faster('100_CH2_0.continuous');
[DataPFC, ~, ~] = load_open_ephys_data_faster('100_CH53_0.continuous');
% extracting the sampling frequency of the data
SamplingFreq = HeadingData.header.sampleRate;       % Sampling frequency of the data
% Downsample the data to different sampling rates for fast processing
TargetSampling = 1250;                             % The goal sampling rate
timesDownSamp  = SamplingFreq / TargetSampling;   % Number of times of downsample the data
lfpPFCDown = decimate(DataPFC,timesDownSamp,'FIR');
lfpHPCDown = decimate(DataHPC,timesDownSamp,'FIR');
timVect = linspace(0,numel(lfpPFCDown)/TargetSampling,numel(lfpPFCDown));

%% Broadband slow wave or delta band

DeltaBandPFC = compute_delta_buzsakiMethod(lfpPFCDown,timVect,TargetSampling,'DeltaBandPFCMat');
DeltaBandHPC = compute_delta_buzsakiMethod(lfpHPCDown,timVect,TargetSampling,'DeltaBandHPCMat');
%% Narrowband Theta wave

ThetaBandPFC = compute_theta_buzsakiMethod(lfpPFCDown,timVect,TargetSampling,'ThetaBandPFCMat');
ThetaBandHPC = compute_theta_buzsakiMethod(lfpHPCDown,timVect,TargetSampling,'ThetaBandHPCMat');
%% beta wave


%% EMGlike Signal
samplingFrequencyEMG = 5;
smoothWindowEMG = 10;

EMGFromLFP = compute_emg_buzsakiMethod(samplingFrequencyEMG, TargetSampling, lfpPFCDown, lfpHPCDown, smoothWindowEMG,'EMGLikeSignalMat');

prEMGtime = DeltaBandPFC.timestamps<EMGFromLFP.timestamps(1) | DeltaBandPFC.timestamps>EMGFromLFP.timestamps(end);
DeltaBandPFC.data(prEMGtime) = []; 
DeltaBandHPC.data(prEMGtime) = [];
ThetaBandPFC.data(prEMGtime) = [];
ThetaBandHPC.data(prEMGtime) = []; 
DeltaBandPFC.timestamps(prEMGtime) = [];

%interpolate to FFT time points;
EMG = interp1(EMGFromLFP.timestamps,EMGFromLFP.smoothed,DeltaBandPFC.timestamps,'nearest');

%Min/Max Normalize
EMG = bz_NormToRange(EMG,[0 1]);

%% Combining and saving the feature matrix
matfilename = 'LFPBuzFeatures';
lfpFeatures = zeros(length(EMG),7);
lfpFeatures(1,:) = DeltaBandPFC.data;
lfpFeatures(2,:) = DeltaBandPFC.data;
lfpFeatures(3,:) = ThetaBandPFC.data;
lfpFeatures(4,:) = ThetaBandHPC.data;
lfpFeatures(5,:) = 
lfpFeatures(6,:) = 
lfpFeatures(7,:) = EMG;