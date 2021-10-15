data = load('data.mat');

d = data.d;
stage_idx = data.epochsLinked(:,3);

% Separating stages index
rem_idx = find(stage_idx == 3);
nrem_idx = find(stage_idx == 2);
wake_idx = find(stage_idx == 1);

% Getting preprocessed feature values
features = ["Delta_Theta","Delta_Alpha","Delta_Beta","Delta_Gamma","Theta_Alpha",...
    "Theta_Beta","Theta_Gamma","Alpha_Beta","Alpha_Gamma","Beta_Gamma","EMG"];

% Separating features into three stages and plotting the histogram
for idx = 1:11
    feat_rem = d(rem_idx,idx);
    feat_nrem = d(nrem_idx,idx);
    feat_wake = d(wake_idx,idx);
    
    % Plotting the histogram
    figure(idx)
    histogram(feat_rem)
    hold on
    histogram(feat_nrem)
    hold on
    histogram(feat_wake)
    legend('REM','NREM','WAKE')
    title(strcat(features(idx)," Histogram"))
    saveas(gcf,strcat(features(idx),"_Histogram.jpg"))
end


