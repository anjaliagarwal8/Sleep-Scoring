function [uniqueStates,inferredStates] = InferStates(visData,variables,states)
% Script written for inferring the unique latent states from a trained mcRBM model.

% Load latent variables
W = variables.W;
VF = variables.VF;
FH = variables.FH;
vb = variables.vb;
hb_cov = variables.hb_cov;
hb_mean = variables.hb_mean;
%% Compute latent activations
% Compute the probabilities of the covariance units (normalize data for
% covariance hidden)
        
dsq = visData.^2;
lsq = sum(dsq);
lsq = lsq./size(visData,2);
lsq = lsq + eps(1);
l = sqrt(lsq);
normD = visData./l;

% Compute logistic_covariance_argument
logisticArg_c = (((FH'*((VF'*normD').^2)).* (-0.5)) + hb_cov)';

% compute hidden_covariance probabilities:
p_hc = sigmoid(logisticArg_c);

% compute logistic_mean_argument (use unnormalised data for mean hidden)
logisticArg_m = visData*W + hb_mean';

% compute hidden_mean probabilities
p_hm = sigmoid(logisticArg_m);

%% Infer latent states from latent activations
p_all = cat(2,p_hc,p_hm);

image_hc = uint8(p_hc.*255.0);
image_hm = uint8(p_hm.*255.0);

% Saving the figures
[status, msg, msgID] = mkdir('latentStates');
cd latentStates
figure1 = figure('visible','off');
image(image_hc)
title('Covariance Hidden Activation')
xlabel('covariance hidden units')
ylabel('Epoch')
saveas(figure1,'covariance_activation.png')

figure2 = figure('visible','off');
image(image_hm)
title('Mean Hidden Activation')
xlabel('Mean hidden units')
ylabel('Epoch')
saveas(figure2,'mean_activation.png')

% Binarize the latent activations
binary_latentActivation = p_all >= 0.5;

save latentStates.mat p_all binary_latentActivation

figure3 = figure('visible','off');
imagesc(binary_latentActivation)
colormap(gray)
title('Binary Latent Activations')
xlabel('Hidden units')
ylabel('Epoch')
saveas(figure3,'binary_activation.png')

% Computing unique latent States, index of unique latent state and count of
% each latent state
[unique_bin,uniqueFramesID,ic] = unique(num2str(binary_latentActivation),'rows');
uniqueAct = binary_latentActivation(uniqueFramesID,:);
uniqueCount = zeros(size(uniqueFramesID));
for i=1:length(uniqueFramesID)
    uniqueCount(i) = length(find(ic==i));
end
p_unique = p_all(uniqueFramesID,:);

fprintf("The number of the unique latent activations is : %d \n",length(uniqueFramesID))

% Check if there are hidden_units that are always off by calculating sum of
% binary hidden state
disp("The sum of the unique latent activations' columns is : ")
sum(uniqueAct,1)

% Saving the above information for further analysis
uniqueStates = zeros(size(uniqueAct,1),size(uniqueAct,2)+2);
inferredStates = [zeros(size(states.downsampledStates,1),1) states.downsampledStates];
for i=1:size(uniqueAct,1)
    uniqueStates(i,1) = i;
    uniqueStates(i,2) = uniqueCount(i);
    uniqueStates(i,3:size(uniqueStates,2)) = uniqueAct(i,:);
    
    RowIdx = find(ismember(binary_latentActivation, uniqueAct(i,:),'rows'));
    inferredStates(RowIdx,1) = i;
end
