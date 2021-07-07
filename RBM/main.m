%% Main program to execute Random Boltzmann Machines

%% Training data

dat = [1,1,1,0,0,0;1,0,1,0,0,0; 1,1,1,0,0,0;0,0,1,1,1,0;0,0,1,1,0,0;0,0,1,1,1,0];
%% Initializing the parameters

n_hid = 2;
n_vis = size(dat,1);
learning_rate = 0.01;
n_epochs = 2000;
gibbs_steps = 1;

%% Initializing the weights and biases for the network

[W,vb,hb] = initialize_weights(n_vis,n_hid);

%% Training the RBM with the data and extracting updated weights and biases with the error 
% on each iteration

[W,vb,hb,error] = train_rbm(dat,W,vb,hb,gibbs_steps,learning_rate,n_epochs);

