%% Method computing the histogram over the desired latent states &
% reordering it for visualization.

load uniqueStates.mat
load obsKeys.mat
load stageDistributionMat.mat

[status, msg, msgID] = mkdir('statesHistogram');
cd statesHistogram

latentStates = length(uniqueStates);
states = uniqueStates(:,1);
frames = uniqueStates(:,2);

bar(states,frames)
xlabel('Latent States')
ylabel('Number of Frames')
saveas(gcf,'statesHistogram.png')

% Compute histogram over latent states of interest
centroidsHist = zeros(latentStates,3);
for s=1:latentStates
    centroidsHist(s,1) = uniqueStates(s,2);
    centroidsHist(s,2) = uniqueStates(s,2);
    centroidsHist(s,3) = s;
end

% Normalize histogram using L2 Norm
centroidsHist(:,2) = centroidsHist(:,2)./norm(centroidsHist(:,2),2);

% Plotting the normalized histogram
cdata = zeros(size(stageMat));
cdata = stageMat./sum(stageMat,2);

% Method re-organizing a matrix according to linkage.
aux_linkage = linkage(cdata,'average','euclidean');
[H,T,outperm] = dendrogram(aux_linkage,0);

colors = cdata(outperm,:);
counts = centroidsHist(outperm,2);

b = bar(states,counts,'CData',colors);


