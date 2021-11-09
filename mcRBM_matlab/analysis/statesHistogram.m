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
