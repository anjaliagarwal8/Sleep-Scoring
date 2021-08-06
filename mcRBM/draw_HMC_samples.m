function [hmc_step, hmc_ave_rej] = draw_HMC_samples(data,negdata,normdata,VF,FH,hb_cov,vb,W,hb_mean,hmc_step,hmc_step_nr,hmc_ave_rej,hmc_target_ave_rej,t1,t2,t3,t4,t5,t6,t7,thresh,feat,featsq,batch_size,feat_mean,len,lengthsq,normcoeff,small,num_vis)

    vel = randn(size(data));
    negdata = data;
    old_energy = zeros(size(data,2));
    gradient = 0;
    normgradient = 
    
    [old_energy,vel] = compute_energy_mcRBM(negdata,vel,VF,FH,hb_cov,vb,W,hb_mean,small);
    gradient = compute_gradient_mcRBM(negdata,VF,FH,hb_cov,vb,W,hb_mean,small);
    
    % Half step
    vel = vel + gradient*(-0.5)*hmc_step;
    negdata = negdata + vel*hmc_step;
    
    % full leap-frog steps
    for ss=1:hmc_step_nr-1
        % Re-evaluate the gradient
        gradient = compute_gradient_mcRBM(negdata,VF,FH,hb_cov,vb,W,hb_mean,small);
        
        % update variables
        vel = vel + gradient*(-hmc_step);
        negdata = negdata + vel*hmc_step;
     
    end
    
    % Final half step
    gradient = compute_gradient_mcRBM(negdata,VF,FH,hb_cov,vb,W,hb_mean,small);
    vel = vel + gradient*(-0.5)*hmc_step;
    
    % compute new energy
    [new_energy,vel] = compute_energy_mcRBM(negdata,vel,VF,FH,hb_cov,vb,W,hb_mean,small);
    
    %Rejection
    thresh = exp(old_energy - new_energy);
    t4 = rand(1,batch_size);
    
end