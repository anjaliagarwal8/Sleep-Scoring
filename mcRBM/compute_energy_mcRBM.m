function [energy,vel] = compute_energy_mcRBM(data,vel,VF,FH,hb_cov,vb,W,hb_mean,small)
    
    num_vis = length(vb);
    % normalize the data
    t6 = data .* data;
    lengthsq = sum(t6);
    energy = lengthsq .* (0.5);
    lengthsq = lengthsq./num_vis + small;
    len = sqrt(lengthsq);
    normcoeff = 1./len;
    normdata  = zeros(size(data));
    for i=1:size(normdata,1)
            normdata(i,:) = data(i,:) .* normcoeff; % Multiply by row
    end
    
    % Covariance contribution
    feat   = dot(VF',normdata);
    featsq = feat .* feat;
    t1 = exp(dot(FH',featsq) .* (-0.5) + hb_cov);
    t2 = -log(t1 + 1);
    energy = energy + sum(t2);
    
    % Mean contribution
    feat_mean = -log(exp(dot(W',data) + hb_mean) + 1);
    energy = energy + sum(feat_mean);
    
    % Visible term
    t6 = zeros(size(data));
    for i=1:size(t6,2)
            t6(:,i) = data(:,i) .* vb; % Multiply by column
    end
    t6 = t6 .* (-1);
    energy = energy + sum(t6);
    
    % kinetic
    t6 = vel.*vel;
    energy = energy + sum(t6).*0.5;
    
    
end