%% Method computing the histogram over the desired latent states &
% reordering it for visualization.

load uniqueStates.mat
load obsKeys.mat
load stageDistributionMat.mat

[status, msg, msgID] = mkdir('statesHistogram');
cd statesHistogram

latentStates = length(uniqueStates);

% Compute histogram over latent states of interest
centroidsHist = zeros(latentStates,3);
for s=1:latentStates
    centroidsHist(s,1) = uniqueStates(s,2);
    centroidsHist(s,2) = uniqueStates(s,2);
    centroidsHist(s,3) = s;
end
