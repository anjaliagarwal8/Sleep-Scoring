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

