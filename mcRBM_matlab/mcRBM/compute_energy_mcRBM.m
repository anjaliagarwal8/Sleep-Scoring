function [energy,vel] = compute_energy_mcRBM(data,vel,VF,FH,hb_cov,vb,W,hb_mean,small,num_vis,store)
    
    
    % normalize the data
    t6 = data .* data;
    lengthsq = sum(t6);
    energy = lengthsq .* 0.5;
    lengthsq = (lengthsq./num_vis) + small;
    len = sqrt(lengthsq);
    normcoeff = 1./len;
    normdata  = data .* normcoeff;
    
    % Covariance contribution
    feat   = VF'*normdata;
    featsq = feat .* feat;
    t1 = exp(((FH'*featsq) .* (-0.5)) + hb_cov);
    t2 = -log(t1 + 1);
    energy = energy + sum(t2);
    
    % Mean contribution
    feat_mean = -log(exp((W'*data) + hb_mean) + 1);
    energy = energy + sum(feat_mean);
    
    % Visible term
    t6 = data .* vb;
    t6 = t6 .* (-1);
    energy = energy + sum(t6);
    
    if store == false
        % kinetic
        t6 = vel.*vel;
    else
        % kinetic
        t6 = data .* data;
    end
    energy = energy + (sum(t6).*0.5);
    
end