%% Main program to execute mean-covariance Restricted Boltzmann Machines (mcRBM)
% Here parameters and data can be initialized

%% Training data


%% Initializing the parameters

n_vis = size(dat,1);
n_epochs = 2000;
startFH = 10; % after these many epochs, update the factor-hidden matrix (wait for visible-factor matrix to converge first)
startwd = 10; % start using L1 weight decay on weight matrices after <startwd> epochs
doPCD = 1; % if doPCD = 0 then use Contrastive Divergence 1, otherwise Persistent Contrastive Divergence 1
epsilon = .075; % learning rate
weightcost_final = 0.001; % L1 weight decay

epsilonVF = 2*epsilon;
epsilonFH = 0.02*epsilon;
epsilonb = 0.02*epsilon;
epsilonw_mean = 0.2*epsilon;
epsilonb_mean = 0.1*epsilon;
    
% mRBM parameters
n_hid_mean = 100; % number of mean hidden units

% cRBM parameters
n_hid_cov = 100; % number of covariance hidden units
num_fac = 256; % number of factors (columns of visible-factor matrix)

% Hybrid Monte Carlo (HMC) parameters
hmc_step_nr = 20; % number of leap-frog steps
hmc_target_ave_rej = 0.1; % target rejection rate
hmc_step =  0.01;
hmc_ave_rej =  hmc_target_ave_rej;

%% Initializing the weights and biases for the network

[W,VF,FH,vb,hb_cov,hb_mean] = initialize_weights(n_vis,n_hid_mean,n_hid_cov,num_fac);

%% Training the RBM with the data and extracting updated weights and biases  

[W,VF,FH,vb,hb_cov,hb_mean,hmc_step, hmc_ave_rej] = train_mcRBM(data,W,VF,FH,vb,hb_cov,hb_mean,batch_size,num_batches,n_vis,num_fac,n_epochs,startFH,startwd,doPCD,epsilonVF,epsilonFH,epsilonb,epsilonw_mean,epsilonb_mean,hmc_step_nr,hmc_target_ave_rej,hmc_step,hmc_ave_rej,weightcost_final);

