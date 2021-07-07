function [v0,h0,vk,hk] = contrastive_divergence(x,W,vb,hb,gibbs_steps)
%%
% Contranstive Divergence
% Calculating the positive and negative divergence to make the two
% probability distributions similar. 

    n_v = length(vb);
    n_h = length(hb);
    v0 = x;
    prob_h0v0 = sigmoid(v0 * W + hb);
    h0 = prob_h0v0 > rand(1,n_h);
    hk = h0;
    
    [prob_hkvk,vk] = gibbs_sampling(hk,W,vb,hb,gibbs_steps);
    hk = prob_hkvk;
    h0 = prob_h0v0;
    
end