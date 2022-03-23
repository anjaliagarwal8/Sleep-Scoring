% 
% load('inferredStates.mat');
% manualscore = inferredStates(:,2);
% latentStates = inferredStates(:,1);
% 
% n=200;
% [status, msg, msgID] = mkdir('HynogramPlots');
% cd HynogramPlots
% states = ["wake","nrem","nrem2rem","rem"];
% latentstates = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];
% for i=1:round(length(latentStates)/n)
%     Hfigure = figure('visible','off');
%     subplot(2,1,1)
%     plot(manualscore((i-1)*n+1:n*i))
%     ylim([1.8,5.2])
%     set(gca,'ytick',[2:5],'yticklabel',states)
%     subplot(2,1,2)
%     plot(latentStates((i-1)*n+1:n*i))
%     ylim([0.8,14.2])
%     set(gca,'ytick',[1:14],'yticklabel',latentstates)
%     saveas(Hfigure,['H',num2str(i),'.png'])
% end
% Hfigure = figure('visible','off');
% subplot(2,1,1)
% plot(manualscore(n*i:length(latentStates)))
% ylim([1.8,5.2])
% set(gca,'ytick',[2:5],'yticklabel',states)
% subplot(2,1,2)
% plot(latentStates(n*i:length(latentStates)))
% ylim([0.8,14.2])
% set(gca,'ytick',[1:14],'yticklabel',latentstates)
% saveas(Hfigure,['H',num2str(i+1),'.png'])
% 

manualscore = inferredStates(:,2);
latentStates = inferredStates(:,1);
sortedLatentStates = zeros(size(latentStates));

states = ["wake","nrem","nrem2rem","rem"];
latentstates = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];
wake_l = [8,12,9,10,1,13,2,14,4,3];
nrem_l = [6,7,5];
rem_l = [11];

for s=1:length(wake_l)
    idx = find(latentStates==wake_l(s));
    sortedLatentStates(idx) = s;
end

for s=1:length(nrem_l)
    idx = find(latentStates==nrem_l(s));
    sortedLatentStates(idx) = s+length(wake_l);
end

for s=1:length(rem_l)
    idx = find(latentStates==rem_l(s));
    sortedLatentStates(idx) = s+length(wake_l)+length(nrem_l);
end

wake_m = find(manualscore==1);
manualscore(wake_m) = 2;

subplot(2,1,1)
plot(manualscore)
ylim([1.8,5.2])
set(gca,'ytick',[2:5],'yticklabel',states)
subplot(2,1,2)
plot(sortedLatentStates)
ylim([0.8,14.2])
set(gca,'ytick',[1:14],'yticklabel',latentstates)