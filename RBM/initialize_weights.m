function [W,vb,hb] = initialize_weights(n_vis,n_hid)
%% 
% Function to initialize the weights and biases of RBM using number 
% of hidden and visible nodes
% Weights are initialized randomly with size (visible_nodes X hidden_nodes)
% while biases are initialized with zeros 
% Note: Weight Initialization can be optimised for better performance

    W = 0.01 * (randn(n_vis, n_hid) - 0.5);
    vb = zeros(1,n_vis); % Visible layer bias
    hb = zeros(1,n_hid); % Hidden layer bias
    
end