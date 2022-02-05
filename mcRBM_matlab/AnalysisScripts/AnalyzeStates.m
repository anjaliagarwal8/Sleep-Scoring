function AnalyzeStates(lfpFeatures,uniqueStates,inferredStates)
%% Method visualizing the boxplots of the LOG initial EEG/EMG 
% data mapping to each latent state.

d = lfpFeatures.lfpFeatures;

% Power Band ratio feature
powerband_features = strings(size(d,2)-1,1);
for f=1:size(d,2)-1
    powerband_features(f) = int2str(f);
end

band_range = [floor(min(d(:,1:size(d,2)-1))); ceil(max(d(:,1:size(d,2)-1)))];
emg_range = [floor(min(d(:,size(d,2)))); ceil(max(d(:,size(d,2))))];

[status, msg, msgID] = mkdir('BoxPlots');
cd BoxPlots

for l=1:length(uniqueStates)
    idx = find(inferredStates(:,1) == l);
    latent_frames = inferredStates(idx,:);
    
    % Detect and remove singletons
    if length(idx) == 1
        continue
    end
    
    %Percentage of main sleep state epochs present in each latent state
    % wake = 1
    % nrem = 3
    % nrem to rem = 4
    % rem = 5
    state_len = zeros();
    len_wake = round((length(find(latent_frames(:,2)==1)))/(length(latent_frames)),3);
    len_nrem = round((length(find(latent_frames(:,2)==3)))/(length(latent_frames)),3);
    len_nremtorem = round((length(find(latent_frames(:,2)==4)))/(length(latent_frames)),3);
    len_rem = round((length(find(latent_frames(:,2)==5)))/(length(latent_frames)),3);
    
    % Extracting Power Band and EMG Values from the epochs belonging to
    % specific latent states 
    dPlotBand = zeros(length(idx),length(powerband_features));
    dPlotEMG = zeros(length(idx),1);
    for f=1:length(powerband_features)
        dPlotBand(:,f) = d(idx,f);
    end
    dPlotEMG(:,1) = d(idx,size(d,2));
    
    %Plotting the Box Plots for Power Bands and EMG Signal for each latent
    %state
    boxfigure = figure('visible','off');
    subplot(1,3,[1,2])
    boxplot(dPlotBand,'Labels',powerband_features)
    ylim([min(band_range(1,:));max(band_range(2,:))])
    ylabel('Log Power')
    title(['LS ',num2str(l),' - ',num2str(length(latent_frames)),' epochs',...
        newline,'WAKE: ',num2str(len_wake*100),'% ,NREM: ',num2str(len_nrem*100),...
        '% ,NREM-REM: ',num2str(len_nremtorem*100),'% ,REM: ',num2str(len_rem*100),'%'])
    
    subplot(1,3,3)
    boxplot(dPlotEMG)
    ylim(emg_range)
    xlabel('EMG')
    
    saveas(boxfigure,['LS',num2str(l),'.png'])
end

cd ../
