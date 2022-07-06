function PlotHypnogram(uniqueStates,inferredStates,LSassignMat,states)

manualStates = inferredStates(:,2);
latentStates = inferredStates(:,1);
nLS = size(uniqueStates,1);
LS = strings(1,nLS);
for i=1:nLS
    LS(1,i) = states.names(1,LSassignMat(i,2))+','+int2str(LSassignMat(i,1));
end

[status, msg, msgID] = mkdir('HypnogramPlot');
cd HypnogramPlot

hypnogram = figure('visible','off');
subplot(2,1,1)
title('Hypnogram')
plot(manualStates)
ylim([states.keys(1)-0.2,states.keys(end)+0.2])
set(gca,'ytick',[states.keys(1):states.keys(end)],'yticklabel',states.names)
subplot(2,1,2)
plot(latentStates)
ylim([0.8,nLS+0.2])
set(gca,'ytick',[1:nLS],'yticklabel',LS)
saveas(hypnogram,['hypnogram','.png'])

cd ../