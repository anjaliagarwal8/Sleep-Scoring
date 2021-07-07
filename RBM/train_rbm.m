function [W,vb,hb,error] = train_rbm(X,W,vb,hb,gibbs_steps,learning_rate,n_epochs)
%%
% Training algorithm of RBM:
% 1. Calculate constrastive divergence using Gibbs Sampling.
% 2. Compute the gradients using the calculated probabilities.
% 3. Update the weights and biases
% 4. Compute the error between original data and reconstructed data.
% 5. Repeat the above steps for specified number of epochs. 

    error = zeros(1,n_epochs);
    n_dat = size(X,1);
    
    for t=1:n_epochs
        er = 0;
        for n=1:n_dat
            [v0,h0,vk,hk] = contrastive_divergence(X(n,:),W,vb,hb,gibbs_steps); %Contrastive Divergence
            [W_grad,vb_grad,hb_grad] = compute_gradients(v0,h0,vk,hk); %Compute gradients
            W = W + learning_rate * W_grad; %Update weights and biases
            vb = vb + learning_rate * vb_grad;
            hb = hb + learning_rate * hb_grad;
            er = er + norm(X(n,:) - vk); %Compute the error
        end 
        error(1,t) = er/n_dat;
    end
end