%% Loading data and variables 
visData = load('visData_final.mat');
variables = load('variables_final.mat');

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
logisticArg_c = (-0.5*FH'*((VF'*normD').^2) + hb_cov)';

% compute hidden_covariance probabilities:
p_hc = sigmoid(logisticArg_c);

% compute logistic_mean_argument (use unnormalised data for mean hidden)
logisticArg_m = d*W + hb_mean';

% compute hidden_mean probabilities
p_hm = sigmoid(logisticArg_m);

%% Infer latent states from latent activations
