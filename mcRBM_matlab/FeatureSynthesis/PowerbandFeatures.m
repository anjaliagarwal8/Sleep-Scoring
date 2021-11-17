%% Synthesis of power band features for each band from the raw sleep dataset

% Frequency ranges for each band
f_all = [0.1 24];
f_delta = [0.5 4];
f_theta = [6 12];
f_beta = [12 23];

% Converting into log space
numfreqs = 100;
FFTfreqs = logspace(log10(f_all(1)),log10(f_all(2)),numfreqs);

Fs = % sampling rate
window = 2;
noverlap = 0;

[thFFTspec,thFFTfreqs,t_FFT] = spectrogram(data,window*Fs,noverlap*Fs,FFTfreqs,Fs);
thFFTspec = (abs(thFFTspec));
[zFFTspec,mu,sig] = zscore(log10(thFFTspec)');
thfreqs = find(thFFTfreqs>=f_theta(1) & thFFTfreqs<=f_theta(2));
thpower = sum((thFFTspec(thfreqs,:)),1);
allpower = sum((thFFTspec),1);

thratio = thpower./allpower;    %Narrowband Theta
    