data = load('data.mat');

d = data.d;
stage_idx = data.epochsLinked(:,3);

% Separating stages index
rem_idx = find(stage_idx == 3);
nrem_idx = find(stage_idx == 2);
wake_idx = find(stage_idx == 1);

% EMG feature values
emg = d(:,11);

% Separating EMG feature into three stages
emg_rem = emg(rem_idx);
emg_nrem = emg(nrem_idx);
emg_wake = emg(wake_idx);

% Plotting the histogram
figure(1)
histogram(emg_rem)
hold on
histogram(emg_nrem)
hold on
histogram(emg_wake)
legend('REM','NREM','WAKE')
title('EMG Histogram')
saveas(gcf,'EMG_Histogram.jpg')