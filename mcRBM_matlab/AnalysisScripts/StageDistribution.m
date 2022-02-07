function [stageMat] = StageDistribution(uniqueStates,inferredStates,states)
%% Compute each latent state's PDF according to how many epochs were manually 
% labeled as Wakefulness, NREM, REM. This can be visualized with an RGB color shade.

% computing the probability of each latent state to belong to each of the 3 sleep stages
latentStates = size(uniqueStates,1);
stageMat = zeros(latentStates,length(states.keys));

for l=1:latentStates
    idx = find(inferredStates(:,1) == l);
    statePopulation = length(idx);
    
    for s=1:length(states.keys)
        stageMat(l,s) = length(find(inferredStates(idx,2)==states.keys(s)));
    end
end

[status, msg, msgID] = mkdir('StageDistribution');
cd StageDistribution

save StageDistributionMat.mat stageMat

%% Plotting the heatmap to visualize latent state stage distribution
cdata = zeros(size(stageMat));
cdata = stageMat./sum(stageMat,2);

% Method re-organizing a matrix according to linkage.
aux_linkage = linkage(cdata,'average','euclidean');
[H,T,outperm] = dendrogram(aux_linkage,0);

heatmapfigure = figure('visible','off');
heatmap(states.names,outperm,cdata(outperm,:),'GridVisible','off');
colormap 'turbo'
ylabel('Latent States')
title('HeatMap')
saveas(heatmapfigure,'heatMap.png')

cd ../