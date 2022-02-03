% function lfpFeatures = GetLFPFeatures(DataHPC,DataPFC,samplingrate,TargetSampling, ...
%     scoredstates)
%% Synthesis of power band features for each band from the raw sleep dataset
[DataHPC, TimeVectLFP, HeadingData] = load_open_ephys_data_faster('100_CH47_0.continuous');
[DataPFC, ~, ~] = load_open_ephys_data_faster('100_CH53_0.continuous');
% extracting the sampling frequency of the data
samplingrate = HeadingData.header.sampleRate;  
% Downsample the data to different sampling rates for fast processing
TargetSampling = 1250;                             % The goal sampling rate
timesDownSamp  = samplingrate / TargetSampling;   % Number of times of downsample the data
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
%% Gamma Band

GammaBandPFC = compute_gamma_buzsakiMethod(lfpPFCDown,timVect,TargetSampling,'GammaBandPFCMat');
GammaBandHPC = compute_gamma_buzsakiMethod(lfpHPCDown,timVect,TargetSampling,'GammaBandHPCMat');
%% EMGlike Signal
samplingFrequencyEMG = 5;
smoothWindowEMG = 10;

EMGFromLFP = compute_emg_buzsakiMethod(samplingFrequencyEMG, TargetSampling, lfpPFCDown, lfpHPCDown, smoothWindowEMG,'EMGLikeSignalMat');

prEMGtime = DeltaBandPFC.timestamps<EMGFromLFP.timestamps(1) | DeltaBandPFC.timestamps>EMGFromLFP.timestamps(end);
DeltaBandPFC.data(prEMGtime) = []; 
DeltaBandHPC.data(prEMGtime) = [];
ThetaBandPFC.data(prEMGtime) = [];
ThetaBandHPC.data(prEMGtime) = []; 
GammaBandPFC.data(prEMGtime) = [];
GammaBandHPC.data(prEMGtime) = [];
DeltaBandPFC.timestamps(prEMGtime) = [];

%interpolate to FFT time points;
EMG = interp1(EMGFromLFP.timestamps,EMGFromLFP.smoothed,DeltaBandPFC.timestamps,'nearest');

%Min/Max Normalize
EMG = bz_NormToRange(EMG,[0 1]);

%% Combining and saving the feature matrix
matfilename = 'LFPBuzFeatures4_long_g';
lfpFeatures = zeros(length(EMG),5);
lfpFeatures(:,1) = DeltaBandPFC.data;
%lfpFeatures(:,2) = DeltaBandHPC.data;
%lfpFeatures(:,3) = ThetaBandPFC.data;
lfpFeatures(:,2) = ThetaBandHPC.data;
lfpFeatures(:,3) = BetaBandPFC.data;
%lfpFeatures(:,6) = BetaBandHPC.data;
lfpFeatures(:,4) = GammaBandHPC.data;
lfpFeatures(:,5) = EMG;

save(matfilename,'lfpFeatures')
%% Plotting the features for further analysis
[status, msg, msgID] = mkdir('FeaturePlots');
cd FeaturePlots
FeaturePlots(DeltaBandPFC,ThetaBandPFC,BetaBandPFC,EMG,'PFC')
FeaturePlots(DeltaBandHPC,ThetaBandHPC,BetaBandHPC,EMG,'HPC')

cd ../
%% Downsampling the scored states to match with the features
States = load('2019-05-21_14-56-02_Post-trial5-states.mat');
%downsampledStates = downsample(States.states,8);
downsampledStates = States.states(1:10837);
save states.mat downsampledStates