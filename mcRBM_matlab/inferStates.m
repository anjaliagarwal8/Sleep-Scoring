% Loading data and variables 
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

% Compute the probabilities of the covariance units
%normalize data for covariance hidden
dsq = np.square(self.d)
		lsq = np.sum(dsq, axis=0)
		lsq /= self.d.shape[1]
		lsq += np.spacing(1)
		l = np.sqrt(lsq)
		normD = self.d/l
		
		# compute logistic_covarinace_argument:	
		logisticArg_c = (-0.5*np.dot(FH.T, np.square(np.dot(VF.T, normD.T))) + bias_cov).T
        
dsq = d.^2;
lsq = sum(dsq);
lsq = lsq./size(d,2);
lsq = lsq + eps(1);
