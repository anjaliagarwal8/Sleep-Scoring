function [W_grad, vb_grad, hb_grad] = compute_gradients(v0, h0, vk, hk)
%%
% Computing the gradients for updating the weights
% 
    W_grad = (v0' * h0) - (vk' * hk);
    vb_grad = (v0 - vk);
    hb_grad = (h0 - hk);
end

