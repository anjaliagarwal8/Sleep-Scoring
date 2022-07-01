function ComputeTransitions(uniqueStates,inferredStates,nStates)
%% Script written for finding the transition probability of each latent state

countTransMat_L = zeros(size(uniqueStates,1),size(uniqueStates,1));
countTransMat_M = zeros(nStates,nStates);

%% Computing transition probability matrix of latent states
for i=1:length(inferredStates)-1
    %if states(i+1,2) - states(i,2) == 1
        a = inferredStates(i,1);
        b = inferredStates(i+1,1);
        countTransMat_L(a,b) = countTransMat_L(a,b) + 1;
    %end
end

%% Computing transition probability matrix of manually scored states
for i=1:length(inferredStates)-1
    %if states(i+1,2) - states(i,2) == 1
        a = inferredStates(i,2);
        b = inferredStates(i+1,2);
        countTransMat_M(a,b) = countTransMat_M(a,b) + 1;
    %end
end

[status, msg, msgID] = mkdir('transMat');
cd transMat

save countTransMat_LS.mat countTransMat_L
save countTransMat_Manual.mat countTransMat_M

transMat_L = countTransMat_L./sum(countTransMat_L,2);
idx_nan = isnan(transMat_L);
transMat_L(idx_nan) = 0;

transMat_M = countTransMat_M./sum(countTransMat_M,2);
idx_nan = isnan(transMat_M);
transMat_M(idx_nan) = 0;

save transitionsMat_LS.mat transMat_L
save transitionMat_Manual.mat transMat_M
%% Detect & remove singletons
threshold = 1; % This threshold can be changed depending on which desired threshold for occurence of latent states
idx = find(uniqueStates(:,2) <= threshold);
transMat_L(idx,:) = [];
transMat_L(:,idx) = [];

thresholdTransMat = transMat_L;
save transitionMatThresholded_LS.mat thresholdTransMat

%% Plotting the directed graph for visualizing the latent states transitions

% G = digraph(countTransMat);
% plot(G)
% 
% mc = dtmc(transMat,'StateNames',["3","3","3","3","3","3","1","3","1","3","5"]);
% figure;
% graphplot(mc,'ColorEdges',true);
% saveas(gcf,'markovgraph.png')
%% HeatMap
% imagesc(countTransMat)

cd ../