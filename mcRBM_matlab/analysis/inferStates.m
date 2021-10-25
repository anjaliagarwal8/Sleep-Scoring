%% Script written for inferring the latent states from a trained mcRBM model.

%% Loading data and variables 
visData = load('visData.mat');
variables = load('variables.mat');

% Load Data
d = visData.visData;
obsKeys = visData.obsKeys;

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
        
dsq = d.^2;
lsq = sum(dsq);
lsq = lsq./size(d,2);
lsq = lsq + eps(1);
l = sqrt(lsq);
normD = d./l;

% Compute logistic_covariance_argument
logisticArg_c = (((FH'*((VF'*normD').^2)).* (-0.5)) + hb_cov)';

% compute hidden_covariance probabilities:
p_hc = sigmoid(logisticArg_c);

% compute logistic_mean_argument (use unnormalised data for mean hidden)
logisticArg_m = d*W + hb_mean';

% compute hidden_mean probabilities
p_hm = sigmoid(logisticArg_m);

%% Infer latent states from latent activations
p_all = cat(2,p_hc,p_hm);

image_hc = uint8(p_hc.*255.0);
image_hm = uint8(p_hm.*255.0);

% Saving the figures
[status, msg, msgID] = mkdir('latentStatesPlots');
cd latentStatesPlots
image(image_hc)
title('Covariance Hidden Activation')
xlabel('covariance hidden units')
ylabel('Epoch')
saveas(gcf,'covariance_activation.png')

image(image_hm)
title('Mean Hidden Activation')
xlabel('Mean hidden units')
ylabel('Epoch')
saveas(gcf,'mean_activation.png')

% Binarize the latent activations
binary_latentActivation = p_all >= 0.5;

save latentStates.mat p_all binary_latentActivation

imagesc(binary_latentActivation)
colormap(gray)
title('Binary Latent Activations')
xlabel('Hidden units')
ylabel('Epoch')
saveas(gcf,'binary_activation.png')

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
uniqueStates = zeros(