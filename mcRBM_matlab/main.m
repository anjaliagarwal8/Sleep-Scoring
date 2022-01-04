clear
clc
%% Main program to execute mean-covariance Restricted Boltzmann Machines (mcRBM)
% Here parameters and data can be initialized

%% Training data

load 'PreprocessedFeatures.mat';
d = PreprocessedFeatures;
%d = sampleData.d;
%obsKeys = sampleData.epochsLinked;
%epochTime = sampleData.epochTime;

totnumcases = size(d,1);
batch_size = totnumcases;
% extracts a subset of the rows of the data matrix. The objective is 
% obtaining a data matrix which can be divided in batches of the 
% selected size with no row left out
d = d(1:(floor(totnumcases/batch_size)*batch_size),:);
obsKeys = obsKeys(1:(floor(totnumcases/batch_size)*batch_size),:);
epochTime = epochTime(1:(floor(totnumcases/batch_size)*batch_size),:);
%% preprocess

dMinRow = min(d);
dMaxRow = max(d);
data = 10.*((d - dMinRow) ./ (dMaxRow - dMinRow) - 0.5);
visData = data;
save visData.mat visData obsKeys epochTime

permIdx = randperm(size(data,1));
data = data(permIdx,:);

%% Initializing the parameters

load input_configuration
num_epochs = 10000;

totnumcases = size(data,1);
num_vis =  size(data,2);       
num_batches = totnumcases/batch_size;

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

