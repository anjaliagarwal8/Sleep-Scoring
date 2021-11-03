%% Compute each latent state's PDF according to how many epochs were manually 
% labeled as Wakefulness, NREM, REM. This can be visualized with an RGB color shade.

load obsKeys.mat
load uniqueStates.mat

% computing the probability of each latent state to belong to each of the 3 sleep stages
latentStates = length(uniqueStates);

for l=1:latentStates
    idx = find(obsKeys(:,1) == l);
    statePopulation = length(idx);
    length_wake = 
end