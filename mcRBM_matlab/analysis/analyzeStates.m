%% Method visualizing the boxplots of the LOG initial EEG/EMG 
% data mapping to each latent state.
load data.mat
load uniqueStates.mat
load obsKeys.mat

powerband_features = ["delta_theta","delta_alpha","delta_beta","delta_gamma","theta_alpha","theta_beta",...
    "theta_gamma","alpha_beta","alpha_gamma","beta_gamma"];
band_range = [floor(min(d(:,1:10))); ceil(max(d(:,1:10)))];
emg_range = [floor(min(d(:,11))); ceil(max(d(:,11)))];

[status, msg, msgID] = mkdir('BoxPlots');
cd BoxPlots

for l=1:length(uniqueStates)
    idx = find(obsKeys(:,1) == l);
    latent_frames = obsKeys(idx,:);
    
    len_wake = round((length(find(latent_frames(:,4)==1)))/(length(latent_frames)),3);
    len_nrem = round((length(find(latent_frames(:,4)==2)))/(length(latent_frames)),3);
    len_rem = round((length(find(latent_frames(:,4)==3)))/(length(latent_frames)),3);
    
    dPlotBand = zeros(length(idx),length(powerband_features));
    dPlotEMG = zeros(length(idx),1);
    for f=1:length(powerband_features)
        dPlotBand(:,f) = d(idx,f);
    end
    dPlotEMG(:,1) = d(idx,11);
    
end