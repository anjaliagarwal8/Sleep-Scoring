%% Compute each latent state's PDF according to how many epochs were manually 
% labeled as Wakefulness, NREM, REM. This can be visualized with an RGB color shade.

load inferredStates.mat
load uniqueStates.mat

% computing the probability of each latent state to belong to each of the 3 sleep stages
latentStates = size(uniqueStates,1);
stageMat = zeros(latentStates,4);

for l=1:latentStates
    idx = find(states(:,1) == l);
    statePopulation = length(idx);
    
    length_wake = length(find(states(idx,2)==1));
    length_nrem = length(find(states(idx,2)==3));
    length_nremtorem = length(find(states(idx,2)==4));
    length_rem = length(find(states(idx,2)==5));
    
    stageMat(l,1) = length_wake;
    stageMat(l,2) = length_nrem;
    stageMat(l,3) = length_nremtorem;
    stageMat(l,4) = length_rem;
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

heatmap({'WAKE','NREM','NREM-REM','REM'},outperm,cdata(outperm,:),'GridVisible','off');
colormap 'turbo'
ylabel('Latent States')
title('HeatMap')
saveas(gcf,'heatMap.png')

