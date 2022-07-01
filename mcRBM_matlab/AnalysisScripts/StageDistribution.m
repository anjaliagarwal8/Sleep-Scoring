function [sor_stageMat,sor_uniqueStates,sor_inferredStates] = StageDistribution(uniqueStates,inferredStates,states)
%% Compute each latent state's PDF according to how many epochs were manually 
% labeled as Wakefulness, NREM, REM. This can be visualized with an RGB color shade.

% computing the probability of each latent state to belong to each of the 3 sleep stages
latentStates = size(uniqueStates,1);
stageMat = zeros(latentStates,length(states.keys));
sor_inferredStates = zeros(size(inferredStates));
sor_uniqueStates = zeros(size(uniqueStates));

for l=1:latentStates
    idx = find(inferredStates(:,1) == l);
    statePopulation = length(idx);
    
    for s=1:length(states.keys)
        stageMat(l,s) = length(find(inferredStates(idx,2)==states.keys(s)));
    end
end

%% Plotting the heatmap to visualize latent state stage distribution
cdata = zeros(size(stageMat));
cdata = stageMat./sum(stageMat,2);

% Method re-organizing a matrix according to linkage.
% aux_linkage = linkage(cdata,'average','euclidean');
% [H,T,outperm] = dendrogram(aux_linkage,0);

%% Sorting latent states based on probability
A_sor = sortrows(cdata,'descend');
thr_val = find(A_sor(:,1) < 0.5,1,'first');
trans_mat = sortrows(A_sor(thr_val:end,:),2,'descend');
A_sor(thr_val:end,:) = trans_mat;

[~,i_sor] = ismember(A_sor,cdata);
sorted_index = i_sor(:,1);

for i=1:size(uniqueStates,1)
    sor_uniqueStates(i,1) = find(sorted_index==uniqueStates(i,1));
    sor_uniqueStates(i,2:end) = uniqueStates(i,2:end);
end
for j=1:size(inferredStates,1)
    sor_inferredStates(j,1) = find(sorted_index==inferredStates(j,1));
    sor_inferredStates(j,2) = inferredStates(j,2);
end

save uniqueStates.mat sor_uniqueStates 
save inferredStates.mat sor_inferredStates

cd ../

sor_stageMat = stageMat(sorted_index,:);

[status, msg, msgID] = mkdir('StageDistribution');
cd StageDistribution

save StageDistributionMat.mat sor_stageMat

xla = [1:size(A_sor,1)];
heatmapfigure = figure('visible','off');
heatmap(states.names,xla,cdata(sorted_index,:),'GridVisible','off');
colormap 'turbo'
ylabel('Latent States')
title('HeatMap')
saveas(heatmapfigure,'heatMap.png')

cd ../