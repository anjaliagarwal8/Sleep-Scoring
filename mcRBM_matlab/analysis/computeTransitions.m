%% Script written for finding the transition probability of each latent state

% Getting the obsKeys data from the inferStates script
load inferredStates.mat
load uniqueStates.mat

countTransMat = zeros(size(uniqueStates,1),size(uniqueStates,1));

%% Computing transition probability matrix
for i=1:length(states)-1
    %if states(i+1,2) - states(i,2) == 1
        a = states(i,1);
        b = states(i+1,1);
        countTransMat(a,b) = countTransMat(a,b) + 1;
    %end
end

[status, msg, msgID] = mkdir('transMat');
cd transMat

save countTransMat.mat countTransMat

transMat = countTransMat./sum(countTransMat,2);
idx_nan = isnan(transMat);
transMat(idx_nan) = 0;

save transitionsMat.mat transMat

%% Detect & remove singletons
threshold = 1; % This threshold can be changed depending on which desired threshold for occurence of latent states
idx = find(uniqueStates(:,2) <= threshold);
transMat(idx,:) = [];
transMat(:,idx) = [];

thresholdTransMat = transMat;
save transitionMatThresholded.mat thresholdTransMat

%% Plotting the directed graph for visualizing the latent states transitions

G = digraph(countTransMat);
plot(G)

mc = dtmc(transMat,'StateNames',["3","1/3","3","3","3","1","3","3","3","3","3/5"]);
figure;
graphplot(mc);
saveas(gcf,'markovgraph.png')
%% HeatMap
imagesc(countTransMat)
