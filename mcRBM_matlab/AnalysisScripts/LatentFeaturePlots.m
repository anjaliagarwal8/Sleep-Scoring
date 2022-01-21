load LFPBuzFeatures4.mat;
load inferredStates.mat;

for i=1:7
    idx = states(:,1) == i;
    features = lfpFeatures(idx,:);
    fig = figure('visible','off');

    subplot(2,2,1)
    plot(features(:,1))
    xlabel('Epoch')
    ylabel('DeltaPFC')
    ylim([0 1])

    subplot(2,2,2)
    plot(features(:,2))
    xlabel('Epoch')
    ylabel('ThetaHPC')
    ylim([0 1])

    subplot(2,2,3)
    plot(features(:,3))
    xlabel('Epoch')
    ylabel('BetaPFC')
    ylim([0 1])

    subplot(2,2,4)
    plot(features(:,4))
    xlabel('Epoch')
    ylabel('EMG')
    ylim([0 1])

    sgtitle(['LS',num2str(i)])
    saveas(fig,['LS',num2str(i)],'jpeg')
end 