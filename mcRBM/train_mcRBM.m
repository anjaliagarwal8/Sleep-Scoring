function [W,vb,hb,error] = train_mcRBM(data,W,vb,hb_mean,hb_cov,learning_rate,n_epochs)
%%
% Training algorithm of RBM:
% 1. Calculate constrastive divergence using Gibbs Sampling.
% 2. Compute the gradients using the calculated probabilities.
% 3. Update the weights and biases
% 4. Compute the error between original data and reconstructed data.
% 5. Repeat the above steps for specified number of epochs. 

    error = zeros(1,n_epochs);
    n_dat = size(X,1);
    small = 0.5;
    
    for t=1:n_epochs
        % Anneal learning rates
        epsilonVFc    = epsilonVF/max(1,t/20);
        epsilonFHc    = epsilonFH/max(1,t/20);
        epsilonbc    = epsilonb/max(1,t/20);
        epsilonw_meanc = epsilonw_mean/max(1,t/20);
        epsilonb_meanc = epsilonb_mean/max(1,t/20);
        weightcost = weightcost_final;
        
        if t <= startFH
            epsilonFHc = 0;
        end
        if t <= startwd	
            weightcost = 0;
        end
        
        % Normalize the data
        t6 = data .* data;
        lengthsq = sum(t6)./length(vb) + small;
        len = sqrt(lengthsq);
        normcoeff = 1./len;
        normdata  = zeros(size(data));
        for i=1:size(normdata,1)
            normdata(i,:) = data(i,:) .* normcoeff; % Multiply by row
        end
        
        % Covariance part
        feat   = dot(VF',normdata);
        featsq = feat .* feat;
        t1 = dot(FH',featsq) .* (-0.5) + hb_cov;
        t2     = sigmoid(t1);
        FHinc  = dot(featsq,t2');
        t3     = dot(FH,t2) .* feat;
        VFinc  = dot(normdata,t3');
        bias_covinc = sum(t2,2) .* (-1);
        
        % Visible bias
        bias_visinc = sum(data,2) .* (-1);
        
        % Mean part
        feat_mean = sigmoid(dot(W',data) + hb_mean) .* (-1);
        W_meaninc = dot(data,feat_mean');
        bias_meaninc = sum(feat_mean,2);
        
        % HMC Sampling
        negdata = randn(size(data));
        vel     = randn(size(data));
        
        if doPCD == 0
            hmc_step, hmc_ave_rej = draw_HMC_samples(data,negdata,normdata,VF,FH,hb_cov,vb,W,hb_mean,hmc_step,hmc_step_nr,hmc_ave_rej,hmc_target_ave_rej,t1,t2,t3,t4,t5,t6,t7,thresh,feat,featsq,batch_size,feat_mean,len,lengthsq,normcoeff,small,num_vis);
        else
            negdataini = negdata;
            hmc_step, hmc_ave_rej = draw_HMC_samples(negdataini,negdata,normdata,VF,FH,hb_cov,vb,W,hb_mean,hmc_step,hmc_step_nr,hmc_ave_rej,hmc_target_ave_rej,t1,t2,t3,t4,t5,t6,t7,thresh,feat,featsq,batch_size,feat_mean,len,lengthsq,normcoeff,small,num_vis);
                
        end
        
    end
end