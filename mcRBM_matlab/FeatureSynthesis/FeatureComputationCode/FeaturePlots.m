function FeaturePlots(DeltaBand, ThetaBand, BetaBand, EMG, channelname)
% function to plot the spectrogram of powerband features and time domain
% plot of the features.
% DeltaBand, ThetaBand, BetaBand, GammaBand --> powerband features in the struc format
% with values timestamps, spec, freqs, data
% EMG --> smoothed EMG signal 
% channelname --> channel which is being used to compute the features (PFC
% or HPC)

[zFFTspec,mu,sig] = zscore(log10(DeltaBand.spec)');
[~,mu_th,sig_th] = zscore((ThetaBand.spec)');
[~,mu_b,sig_b] = zscore((BetaBand.spec)');
viewwin  =[DeltaBand.timestamps(1) DeltaBand.timestamps(end)];
clusterfig = figure('visible','off');
	subplot(9,1,[1:2])
    imagesc(DeltaBand.timestamps,log2(DeltaBand.freqs),log10(DeltaBand.spec))
        axis xy
        set(gca,'YTick',(log2([1 2 4 8 16 32 64 128])))
        set(gca,'YTickLabel',{'1','2','4','8','16','32','64','128'})
        caxis([3.5 6.5])
        caxis([min(mu)-2*max(sig) max(mu)+2*max(sig)])
        xlim(viewwin)
        colorbar('east')
        ylim([log2(DeltaBand.freqs(1)) log2(DeltaBand.freqs(end))+0.2])
        set(gca,'XTickLabel',{})
        ylabel({'swLFP','f (Hz)'})
        title(['Feature Plot: ',channelname]);
	subplot(9,1,3)
        imagesc(DeltaBand.timestamps,log2(ThetaBand.freqs),log10(ThetaBand.spec))
        axis xy
        set(gca,'YTick',(log2([1 2 4 8 16 32 64 128])))
        set(gca,'YTickLabel',{'1','2','4','8','16','32','64','128'})
        %caxis([3.5 6.5])
        caxis([min(mu_th)-2*max(sig_th) max(mu_th)+2*max(sig_th)])
        xlim(viewwin)
        %colorbar('east')
        ylim([log2(ThetaBand.freqs(1)) log2(ThetaBand.freqs(end))+0.2])
        ylabel({'thLFP','f (Hz)'})
        set(gca,'XTickLabel',{})
        
    subplot(9,1,4)
        imagesc(DeltaBand.timestamps,log2(BetaBand.freqs),log10(BetaBand.spec))
        axis xy
        set(gca,'YTick',(log2([1 2 4 8 16 32 64 128])))
        set(gca,'YTickLabel',{'1','2','4','8','16','32','64','128'})
        %caxis([3.5 6.5])
        caxis([min(mu_b)-2*max(sig_b) max(mu_b)+2*max(sig_b)])
        xlim(viewwin)
        %colorbar('east')
        ylim([log2(BetaBand.freqs(1)) log2(BetaBand.freqs(end))+0.2])
        ylabel({'BetaLFP','f (Hz)'})
        set(gca,'XTickLabel',{})
        
   	subplot(7,1,4)
        hold on
        plot(DeltaBand.timestamps,DeltaBand.data,'k')
        %plot(synchtimes',thresh*ones(size(synchtimes))','r')
        ylabel('SW')
        box on
        ylim([0 1])
        xlim(viewwin)
        set(gca,'XTickLabel',{})
        
   	subplot(7,1,5)
        hold on
        plot(DeltaBand.timestamps,ThetaBand.data,'k')
        %plot(synchtimes',thresh*ones(size(synchtimes))','r')
        ylabel('Theta')
        box on
        ylim([0 1])
        xlim(viewwin)
        set(gca,'XTickLabel',{})

    subplot(7,1,6)
        hold on
        plot(DeltaBand.timestamps,BetaBand.data,'k')
        %plot(synchtimes',thresh*ones(size(synchtimes))','r')
        ylabel('Beta')
        box on
        ylim([0 1])
        xlim(viewwin)
        set(gca,'XTickLabel',{})
        
   	subplot(7,1,7)
        hold on
        plot(DeltaBand.timestamps,EMG,'k')
        %plot(synchtimes',thresh*ones(size(synchtimes))','r')
        ylabel('EMG')
        box on
        ylim([0 1])
        xlim(viewwin)
        xlabel('t (s)')
        
	saveas(clusterfig,['FeaturePlot',channelname],'jpeg')