latentstates = inferredStates(:,1);
FeatureMat = zeros(size(lfpFeatures,2),size(uniqueStates,1));

for l=1:size(uniqueStates,1)
    idx = latentstates == l;
    FeatureMat(:,l) = mean(lfpFeatures(idx,:));
end

figure
features = {'Delta-PFC','Theta-HPC','Beta-PFC','Gamma-HPC','EMG-like'};
states = uint32(1):uint32(size(uniqueStates,1));
h=heatmap(states,features,FeatureMat,'Colormap',hot,'ColorLimits',[0 1]);
h.Title = 'Feature Strength';
h.XLabel = 'Latent States';
h.YLabel = 'Features';

