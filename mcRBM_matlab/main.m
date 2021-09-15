%% Main program to execute mean-covariance Restricted Boltzmann Machines (mcRBM)
% Here parameters and data can be initialized

%% Training data

sampleData = load('sampleData.mat');
d = sampleData.d;
obsKeys = sampleData.epochsLinked;
epochTime = sampleData.epochTime;

%% preprocess

dMinRow = min(d);
dMaxRow = max(d);
data = 10.*((d - dMinRow) ./ (dMaxRow - dMinRow) - 0.5);
visData = data;
save visData.mat visData obsKeys epochTime
%% Initializing the parameters

load input_configuration
num_epochs = 1000;
batch_size = 256;
num_fac = 11;
num_hid_cov = 11;
num_hid_mean = 10;
epsilon = .01;

totnumcases = size(data,1);
%data = data.whitendata(1:floor(totnumcases/batch_size)*batch_size,:);
%totnumcases = size(data,1);

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
num_epochs = 50;
data = data';
[W,VF,FH,vb,hb_cov,hb_mean,hmc_step, hmc_ave_rej] = train_mcRBM(data,W,VF,FH,vb,hb_cov,hb_mean,batch_size,num_batches,num_vis,num_fac,num_epochs,startFH,startwd,doPCD,epsilonVF,epsilonFH,epsilonb,epsilonw_mean,epsilonb_mean,hmc_step_nr,hmc_target_ave_rej,hmc_step,hmc_ave_rej,weightcost_final,apply_mask);

