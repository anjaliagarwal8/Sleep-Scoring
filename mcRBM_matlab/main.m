clear
clc

%% Main program to execute mean-covariance Restricted Boltzmann Machines (mcRBM)

[status, msg, msgID] = mkdir('AnalysisResults');
cd AnalysisResults
%% Training data

lfpFeatures = load('LFPBuzFeatures4_long_g.mat');
states = load('states.mat');
features = {'Delta-PFC','Theta-HPC','Beta-PFC','Gamma-HPC','EMG-like'};

d = lfpFeatures.lfpFeatures;

totnumcases = size(d,1);
batch_size = totnumcases;
% extracts a subset of the rows of the data matrix. The objective is 
% obtaining a data matrix which can be divided in batches of the 
% selected size with no row left out
d = d(1:(floor(totnumcases/batch_size)*batch_size),:);
%obsKeys = obsKeys(1:(floor(totnumcases/batch_size)*batch_size),:);
%epochTime = epochTime(1:(floor(totnumcases/batch_size)*batch_size),:);
%% preprocess

dMinRow = min(d);
dMaxRow = max(d);
data = 10.*((d - dMinRow) ./ (dMaxRow - dMinRow) - 0.5);
visData = data;
save visData.mat visData 

permIdx = randperm(size(data,1));
data = data(permIdx,:);

%% Initializing the parameters

load input_configuration
num_epochs = 10000;

batch_size = totnumcases;
totnumcases = size(data,1);
num_vis =  size(data,2);       
num_batches = totnumcases/batch_size;
num_fac = size(data,2);
num_hid_cov = size(data,2);
num_hid_mean = size(data,2) - 1;

% training parameters
epsilonVF = 2*epsilon;
epsilonFH = 0.02*epsilon;
epsilonb = 0.02*epsilon;
epsilonw_mean = 0.2*epsilon;
epsilonb_mean = 0.1*epsilon;

% Hybrid Monte Carlo (HMC) parameters
hmc_step =  0.01;
hmc_ave_rej =  hmc_target_ave_rej;

%% Initializing the weights and biases for the network

[W,VF,FH,vb,hb_cov,hb_mean] = initialize_weights(num_vis,num_hid_mean,num_hid_cov,num_fac);

%% Training the RBM with the data and extracting updated weights and biases  

data = data';
[W,VF,FH,vb,hb_cov,hb_mean,hmc_step, hmc_ave_rej] = train_mcRBM(data,W,VF,FH,vb,hb_cov,hb_mean,batch_size,num_batches,num_vis,num_fac,num_epochs,startFH,startwd,doPCD,epsilonVF,epsilonFH,epsilonb,epsilonw_mean,epsilonb_mean,hmc_step_nr,hmc_target_ave_rej,hmc_step,hmc_ave_rej,weightcost_final,apply_mask);
variables.W = W; %Weight 
variables.VF = VF; %Visible Factor
variables.FH = FH; %Factor Hidden
variables.vb = vb; %visible bias
variables.hb_cov = hb_cov; %hidden covariance bias
variables.hb_mean = hb_mean; %hidden mean bias

%% Analysis of the latent states and final weights

[uniqueStates,inferredStates] = InferStates(visData,variables,states);
[stageMat,sor_uniqueStates,sor_inferredStates] = StageDistribution(uniqueStates,inferredStates,states);
AnalyzeStates(lfpFeatures,sor_uniqueStates,sor_inferredStates,states);
[LSassignMat] = StatesHistogram(sor_uniqueStates,sor_inferredStates,stageMat);
ComputeTransitions(sor_uniqueStates,sor_inferredStates,states);
PlotHypnogram(sor_uniqueStates,sor_inferredStates,LSassignMat,states);
AnalyzeFeatures(lfpFeatures,sor_uniqueStates,sor_inferredStates, features);