function StatesHistogram(uniqueStates,inferredStates,stageMat)
%% Method computing the histogram over the desired latent states &
% reordering it for visualization.

[status, msg, msgID] = mkdir('statesHistogram');
cd statesHistogram

truestates = uniqueStates(:,1);
frames = uniqueStates(:,2);

histfigure = figure('Visible','off');
bar(truestates,frames)
xlabel('Latent States')
ylabel('Number of Frames')
saveas(histfigure,'statesHistogram.png')

% Compute histogram over latent states of interest
% centroidsHist = zeros(latentStates,3);
% for s=1:latentStates
%     centroidsHist(s,1) = uniqueStates(s,2);
%     centroidsHist(s,2) = uniqueStates(s,2);
%     centroidsHist(s,3) = s;
% end
% 
% % Normalize histogram using L2 Norm
% centroidsHist(:,2) = centroidsHist(:,2)./norm(centroidsHist(:,2),2);
% 
% % Plotting the normalized histogram
% cdata = zeros(size(stageMat));
% cdata = stageMat./sum(stageMat,2);
% 
% % Method re-organizing a matrix according to linkage.
% aux_linkage = linkage(cdata,'average','euclidean');
% [H,T,outperm] = dendrogram(aux_linkage,0);
% 
% colors = cdata(outperm,:);
% counts = centroidsHist(outperm,2);

% b = bar(truestates,counts,'FaceColor','flat');
% b.CData = colors;
% xlabel('Latent States');
% ylabel('L2 Normalized Count');
% title('Histogram over the latent states');
% text(latentStates-20,max(counts),'Wake','FontSize',14,'Color',[1 0 0])
% text(latentStates-20,max(counts)-0.03,'NREM','FontSize',14,'Color',[0 1 0])
% text(latentStates-20,max(counts)-0.06,'REM','FontSize',14,'Color',[0 0 1])
% saveas(gcf,'coloredStatesHistogram.png')

cd ../