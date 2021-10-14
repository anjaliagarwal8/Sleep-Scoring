data = load('data.mat');

d = data.d;
stage_idx = data.epochsLinked(:,3);

% Separating stages index
rem_idx = find(stage_idx == 3);
nrem_idx = find(stage_idx == 2);
wake_idx = find(stage_idx == 1);

% Log and zero mean EMG feature
dlog = abs(log(d));
emg = dlog(:,11) - mean(dlog(:,11));

% Log and zero mean delta/theta feature


% Separating EMG feature into three stages
emg_rem = emg(rem_idx);
emg_nrem = emg(nrem_idx);
emg_wake = emg(wake_idx);

% Separating delta/theta into three stages
