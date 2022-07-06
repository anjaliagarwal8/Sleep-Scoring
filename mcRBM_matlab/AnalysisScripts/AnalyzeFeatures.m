function AnalyzeFeatures(lfpFeatures,uniqueStates,inferredStates, features)

latentstates = inferredStates(:,1);
d = lfpFeatures.lfpFeatures;
FeatureMat = zeros(size(d,2),size(uniqueStates,1));

for l=1:size(uniqueStates,1)
    idx = latentstates == l;
    FeatureMat(:,l) = mean(d(idx,:));
end

[status, msg, msgID] = mkdir('FeatureAnalysis');
cd FeatureAnalysis

featureanalyze = figure('visible','off');
states = uint32(1):uint32(size(uniqueStates,1));
h=heatmap(states,features,FeatureMat,'Colormap',hot,'ColorLimits',[0 1]);
h.Title = 'Feature Strength';
h.XLabel = 'Latent States';
h.YLabel = 'Features';

saveas(featureanalyze,['FeatureStrengthAnalysis','.png'])

cd ../
