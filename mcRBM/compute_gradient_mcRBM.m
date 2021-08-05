function gradient = compute_gradient_mcRBM(data,VF,FH,hb_cov,vb,W,hb_mean,small)

    num_vis = length(vb);
    
    % normalize input data
    t6 = data .* data;
    lengthsq = sum(t6)./num_vis + small;
    len = sqrt(lengthsq);
    normcoeff = 1./len;
    normdata  = zeros(size(data));
    for i=1:size(normdata,1)
            normdata(i,:) = data(i,:) .* normcoeff; % Multiply by row
    end
    
    feat   = dot(VF',normdata);
    featsq = feat .* feat;
    t1 = dot(FH',featsq) .* (-0.5) + hb_cov;
    t2 = sigmoid(t1);
    t3 = dot(FH,t2) .* feat;
    normgradient = dot(VF,t3);
    
    normcoeff = 1/(len .* lengthsq);
    gradient = normgradient .* data;
    t4 = -sum(gradient)./num_vis;
    
    gradient  = zeros(size(data));
    for i=1:size(gradient,1)
            gradient(i,:) = data(i,:) .* t4; % Multiply by row
    end
    
    t6  = zeros(size(normgradient));
    for i=1:size(t6,1)
            t6(i,:) = normgradient(i,:) .* lengthsq; % Multiply by row
    end
    
    gradient = gradient + t6;
    for i=1:size(gradient,1)
            gradient(i,:) = gradient(i,:) .* normcoeff; % Multiply by row
    end
    gradient = gradient + data; % add quadratic term gradient
    gradient = gradient - vb;   % add visible bias term
    feat_mean = sigmoid(dot(W',data) + hb_mean);
    gradient = gradient - dot(W',feat_mean); % add mean contribution to gradient
    
end