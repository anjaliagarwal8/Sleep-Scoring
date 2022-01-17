%% Synthesis of power band features for each band from the raw sleep dataset

clear; close all; clc
[DataHPC, TimeVectLFP, HeadingData] = load_open_ephys_data_faster('100_CH2.continuous');
[DataPFC, ~, ~] = load_open_ephys_data_faster('100_CH33.continuous');
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

BetaBandPFC = compute_beta_buzsakiMethod(lfpPFCDown,timVect,TargetSampling,'BetaBandPFCMat');
BetaBandHPC = compute_beta_buzsakiMethod(lfpHPCDown,timVect,TargetSampling,'BetaBandHPCMat');
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
matfilename = 'LFPBuzFeatures4';
lfpFeatures = zeros(length(EMG),4);
lfpFeatures(:,1) = DeltaBandPFC.data;
%lfpFeatures(:,2) = DeltaBandHPC.data;
%lfpFeatures(:,3) = ThetaBandPFC.data;
lfpFeatures(:,2) = ThetaBandHPC.data;
lfpFeatures(:,3) = BetaBandPFC.data;
%lfpFeatures(:,6) = BetaBandHPC.data;
lfpFeatures(:,4) = EMG;

save(matfilename,'lfpFeatures')
%% Plotting the features for further analysis
[status, msg, msgID] = mkdir('FeaturePlots');
cd FeaturePlots
FeaturePlots(DeltaBandPFC,ThetaBandPFC,BetaBandPFC,EMG,'PFC')
FeaturePlots(DeltaBandHPC,ThetaBandHPC,BetaBandHPC,EMG,'HPC')

cd ../
%% Downsampling the scored states to match with the features
States = load('post_trial1_2018-02-16_11-34-47-states.mat');
%downsampledStates = downsample(States.states,8);
downsampledStates = States.states(1:2699);
save states.mat downsampledStates