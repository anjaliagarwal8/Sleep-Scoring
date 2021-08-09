function [W,VF,FH,vb,hb_cov,hb_mean] = initialize_weights(n_vis,n_hid_mean,n_hid_cov,num_fac)
%% 
% Function to initialize the weights and biases of mcRBM using number 
% of hidden nodes (covariance and mean) and visible nodes
% Weights are initialized randomly with size (visible_nodes X hidden_nodes)
% while biases are initialized with zeros 
% Note: Weight Initialization can be optimised for better performance

    W = 0.05 * (randn(n_vis, n_hid_mean)); % Mean Weight matrix
    vb = zeros(n_vis,1); % Visible layer bias
    hb_mean = -2.0*ones(n_hid_mean,1); % Mean hidden layer bias
    hb_cov = 2.0*ones(n_hid_cov,1); % Covariance hidden layer bias
    
    VF = 0.02 * (randn(n_vis, num_fac)); % Visible Factor matrix
    FH = eye(num_fac,n_hid_cov); % Factor Hidden matrix
    
end