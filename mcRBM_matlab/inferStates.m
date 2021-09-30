%% Loading data and variables 
sampleData = load('data.mat');
visData = load('visData_final.mat');
variables = load('variables_p.mat');
variables_m = load('variables_final.mat');

% Load Data

d = visData.visData;
obsKeys = visData.obsKeys;

% Load latent variables
W = variables.w_mean;
VF = variables.VF;
FH = variables.FH;
vb = variables.bias_vis;
hb_cov = variables.bias_cov;
hb_mean = variables.bias_mean;

W = variables_m.W;
VF = variables_m.VF;
FH = variables_m.FH;
vb = variables_m.vb;
hb_cov = variables_m.hb_cov;
hb_mean = variables_m.hb_mean;
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

image(image_hc)
colorbar

% Binarize the latent activations
binary_latentActivation = p_all >= 0.5;

save latentStates.mat p_all binary_latentActivation

imagesc(binary_latentActivation)
colormap(gray)