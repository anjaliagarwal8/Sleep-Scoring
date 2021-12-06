function EMGFromLFP = compute_emg_buzsakiMethod(x, y, sig1, sig2, smoothWindow,matfilename)
% The input of this function
% x --> is the target sampling of the EMG
% y --> is the sampling frequency of the data
% sig1 --> the signal of the first channel
% sig2 --> the signal of the second channel
% smooth --> the smoothing window for the EMG data
% matfilename --> the name of the saved matrix
binScootS = 1 ./ x;
binScootSamps = round(y*binScootS);
% 2. You filter the signal in the high frequency range
% Filter first in high frequency band to remove low-freq physiologically
% correlated LFPs (e.g., theta, delta, SPWs, etc.)
maxfreqband = floor(max([625 y/2]));
xcorr_freqband = [275 300 maxfreqband-25 maxfreqband]; % Hz
filteredSig1 = filtsig_in(sig1, y, xcorr_freqband);
filteredSig2  = filtsig_in(sig2, y, xcorr_freqband);
% 3. The relevant parameters important for further correlation analysis
xcorr_window_samps = round(binScootS*y);
xcorr_window_inds = -xcorr_window_samps:xcorr_window_samps;%+- that number of ms in samples
timestamps = (1+xcorr_window_inds(end)):binScootSamps:(size(filteredSig1,1)-xcorr_window_inds(end));
numbins = length(timestamps);
EMGCorr = zeros(numbins, 1);
counter = 1;
c1 = [];
c2 = [];
binind = 0;
binindstart = 1;
% loop through different windows
for i = 1: numel(timestamps)
    binind = binind+1;
    s1 =filteredSig1(timestamps(i) + xcorr_window_inds);
    s2 =filteredSig2(timestamps(i)+ xcorr_window_inds);
    c1 = cat(2,c1,s1);
    c2 = cat(2,c2,s2);
    binindend = binind;
    tmp = corr(c1,c2);
    tmp = diag(tmp);
    EMGCorr(binindstart:binindend) = EMGCorr(binindstart:binindend) + tmp;
    c1 = [];
    c2 = [];
    binindstart = binind+1;
end
EMGCorr = EMGCorr/(2*(2-1)/2);
EMGNorm = bz_NormToRange(EMGCorr,[0 1]);
% Making the final structure of the EMG
EMGFromLFP.timestamps = timestamps'./y;
EMGFromLFP.data = EMGCorr;
EMGFromLFP.Norm = EMGNorm;
EMGFromLFP.channels = 'HPC and PFC';
EMGFromLFP.detectorName = 'bz_EMGFromLFP';
EMGFromLFP.samplingFrequency = x;
% Smoothing the EMG data
smoothfact = smoothWindow;
dtEMG = 1/EMGFromLFP.samplingFrequency;
EMGFromLFP.smoothed = smooth(EMGFromLFP.data,smoothfact/dtEMG,'moving');
% saving the data matrix
save(matfilename,'EMGFromLFP');
end