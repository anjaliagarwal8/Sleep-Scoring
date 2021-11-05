%% Compute each latent state's PDF according to how many epochs were manually 
% labeled as Wakefulness, NREM, REM. This can be visualized with an RGB color shade.

load obsKeys.mat
load uniqueStates.mat

% computing the probability of each latent state to belong to each of the 3 sleep stages
latentStates = length(uniqueStates);
stageMat = zeros(latentStates,3);

for l=1:latentStates
    idx = find(obsKeys(:,1) == l);
    statePopulation = length(idx);
    
    length_wake = length(find(obsKeys(idx,4)==1));
    length_nrem = length(find(obsKeys(idx,4)==2));
    length_rem = length(find(obsKeys(idx,4)==3));
    
    stageMat(l,1) = length_wake;
    stageMat(l,2) = length_nrem;
    stageMat(l,3) = length_rem;
end

[status, msg, msgID] = mkdir('StageDistribution');
cd StageDistribution

save StageDistributionMat.mat stageMat

%% Plotting the heatmap to visualize latent state stage distribution
cdata = zeros(size(stageMat));
cdata = stageMat./sum(stageMat,2);

% Method re-organizing a matrix according to linkage.
aux_linkage = linkage(cdata,'average','euclidean');
D = pdist(lstateColor);
leafOrder = optimalleaforder(aux_linkage,D);
[a,b,c] = dendrogram(aux_linkage,0);


heatmap(lstateColor(leaves,:),'GridVisible','off');
colormap 'turbo'

