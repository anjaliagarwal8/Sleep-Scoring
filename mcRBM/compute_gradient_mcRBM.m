function gradient = compute_gradient_mcRBM(data,VF,FH,hb_cov,vb,W,hb_mean,small)

    num_vis = length(vb);
    
    % normalize input data
    t6 = data .* data;
    lengthsq = sum(t6)./num_vis + small;
    len = sqrt(lengthsq);
    normcoeff = 1./len;
    normdata  = data .* normcoeff;
    
    feat   = dot(VF',normdata);
    featsq = feat .* feat;
    t1 = dot(FH',featsq) .* (-0.5) + hb_cov;
    t2 = sigmoid(t1);
    t3 = dot(FH,t2) .* feat;
    normgradient = dot(VF,t3);
    
    normcoeff = 1/(len .* lengthsq);
    gradient = normgradient .* data;
    t4 = -sum(gradient)./num_vis;
    
    gradient  = data .* t4;
    
    t6  = normgradient .* lengthsq;
    
    gradient = gradient + t6;
    gradient = gradient .* normcoeff;
    gradient = gradient + data; % add quadratic term gradient
    gradient = gradient - vb;   % add visible bias term
    feat_mean = sigmoid(dot(W',data) + hb_mean);
    gradient = gradient - dot(W',feat_mean); % add mean contribution to gradient
    
end