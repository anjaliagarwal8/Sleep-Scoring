function [prob_hkvk,vk] = gibbs_sampling(hk,W,vb,hb,gibbs_steps)
%%
% Gibbs Sampling Algorithm

    for k = 1:gibbs_steps
        prob_vkhk = sigmoid(hk * W' + vb);
        vk = prob_vkhk;
        
        prob_hkvk = sigmoid(vk * W + hb);
        hk = prob_hkvk > rand(1,length(hb));
    end
end