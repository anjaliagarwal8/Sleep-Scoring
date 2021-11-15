load data.mat
load uniqueStates.mat
load obsKeys.mat

powerband_features = ['delta_theta','delta_alpha','delta_beta','delta_gamma','theta_alpha','theta_beta',
    'theta_gamma','alpha_beta','alpha_gamma','beta_gamma'];
band_range = [floor(min(d(:,1:10))); ceil(max(d(:,1:10)))];
emg_range = [floor(min(d(:,11))); ceil(max(d(:,11)))];

for l=1:length(uniqueStates)
    
end