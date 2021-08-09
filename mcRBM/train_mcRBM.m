function [W,VF,FH,vb,hb_cov,hb_mean,hmc_step, hmc_ave_rej] = train_mcRBM(X,W,VF,FH,vb,hb_cov,hb_mean,batch_size,num_batches,num_vis,num_fac,n_epochs,startFH,startwd,doPCD,epsilonVF,epsilonFH,epsilonb,epsilonw_mean,epsilonb_mean,hmc_step_nr,hmc_target_ave_rej,hmc_step,hmc_ave_rej,weightcost_final)
%%


    small = 0.5;
    normVF = 1;
    negdata = randn(num_vis,batch_size);
   
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
        
        for batch=1:num_batches
            
            % get current minibatch
            data = X(batch*batch_size,(batch+1)*batch_size);
            
            % Normalize the data
            t6 = data .* data;
            lengthsq = sum(t6)./num_vis + small;
            len = sqrt(lengthsq);
            normcoeff = 1./len;
            normdata  = zeros(size(data));
            for i=1:size(normdata,1)
                normdata(i,:) = data(i,:) .* normcoeff; % Multiply by row
            end
            
            %% compute positive sample derivatives
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
            
            % HMC sampling: draw an approximate sample from the model
            if doPCD == 0 %  CD-1 (set negative data to current training samples)
                hmc_step, hmc_ave_rej,negdata = draw_HMC_samples(data,VF,FH,hb_cov,vb,W,hb_mean,hmc_step,hmc_step_nr,hmc_ave_rej,hmc_target_ave_rej,batch_size,small,num_vis);
            else % PCD-1 (use previous negative data as starting point for chain)
                negdataini = negdata;
                hmc_step, hmc_ave_rej,negdata = draw_HMC_samples(negdataini,VF,FH,hb_cov,vb,W,hb_mean,hmc_step,hmc_step_nr,hmc_ave_rej,hmc_target_ave_rej,batch_size,small,num_vis);
            end
            
            %% compute derivatives at the negative samples
            % normalize input data
            t6 = negdata .* negdata;
            lengthsq = sum(t6)./num_vis + small;
            len = sqrt(lengthsq);
            normcoeff = 1./len;
            normdata  = zeros(size(negdata));
            for i=1:size(normdata,1)
                normdata(i,:) = negdata(i,:) .* normcoeff; % Multiply by row
            end
            
            % covariance part
            feat = dot(VF',normdata);
            featsq = feat .* feat;
            t1 = dot(FH',featsq) .* (-0.5);
            t1 = t1 + hb_cov;
            t2 = sigmoid(t1);
            FHinc = FHinc - dot(featsq,t2');
            FHinc = FHinc .* 0.5;
            t3 = dot(FH,t2) .* feat;
            VFinc = VFinc - dot(normdata,t3');
            bias_covinc = bias_covinc + sum(t2,2);
            
            % visible bias
            bias_visinc = bias_visinc + sum(negdata,2);
            
            % mean part
            feat_mean = sigmoid(dot(W',negdata) + hb_mean);
            W_meaninc = W_meaninc + dot(negdata,feat_mean');
            bias_meaninc = bias_meaninc + sum(feat_mean,2);
            
            % update parameters
            VFinc = VFinc + sign(VF) .* weightcost;
            VF = VF + VFinc .* (-epsilonVFc/batch_size);
            % normalizing columns of VF
            t8 = VF .* VF;
            t10 = sqrt(sum(t8));
            t5 = sum(t10,2);
            normVF = .95*normVF + (.05/num_fac) * t5(1,1);
            t10 = 1/t10;
            for i=1:size(VF,1)
                VF(i,:) = VF(i,:) .* t10; % Multiply by row
            end
            VF = VF .* normVF;
            hb_cov = hb_cov + bias_covinv .* ( -epsilonbc/batch_size);
            vb = vb + bias_visinc .* ( -epsilonbc/batch_size);
            
            if t>startFH
                FHinc = FHinc + sign(FH) .* weightcost;
            
                FH = FH + FHinc .* (-epsilonFHc/batch_size);
                t9 = FH > 0;
                FH = FH .* t9;
                if apply_mask == 1
                    FH = FH .* mask;
                end
                % normalize columns of FH
                t11 = 1/sum(FH);
                for i=1:size(FH,1)
                    FH(i,:) = FH(i,:) .* t11; % Multiply by row
                end
            end
            W_meaninc = W_meaninc + sign(W) .* weightcost;
            W = W + W_meaninc .* (-epsilonw_meanc/batch_size);
            hb_mean = hb_mean + bias_meaninc .* (-epsilonb_meanc/batch_size);
            
            % Display the parameters
            
        end
    end
end