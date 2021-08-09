function [hmc_step, hmc_ave_rej] = draw_HMC_samples(data,VF,FH,hb_cov,vb,W,hb_mean,hmc_step,hmc_step_nr,hmc_ave_rej,hmc_target_ave_rej,batch_size,small,num_vis)

    vel = randn(size(data));
    negdata = data;
    
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
    t4 = t4 < thresh;
    
    % Update negdata and rejection rate
    t4 = t4*(-1);
    t4 = t4 +  1;
    t5 = sum(t4,2);
    rej = t5(1,1)/batch_size;
    t6  = zeros(num_vis,batch_size);
    for i=1:size(t6,1)
            t6(i,:) = data(i,:) .* t4; % Multiply by row
    end
    t7  = zeros(num_vis,batch_size);
    for i=1:size(t7,1)
            t7(i,:) = negdata(i,:) .* t4; % Multiply by row
    end
    negdata = negdata - t7 + t6;
    hmc_ave_rej = 0.9*hmc_ave_rej + 0.1*rej;
    if hmc_ave_rej < hmc_target_ave_rej
        hmc_step = min(hmc_step*1.01,0.25);
    else
        hmc_step = max(hmc_step*0.99,.001);
    end
    
end