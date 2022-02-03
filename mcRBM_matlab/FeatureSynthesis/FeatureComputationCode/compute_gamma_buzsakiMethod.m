function [GammaBand] = compute_gamma_buzsakiMethod(lfp,timestamps,samplingrate,matfilename)
% Function to compute the Gamma band. It
% computes gamma band using the power spectrum slope
% lfp --> lfp data 
% timestamps --> time vector of given lfp data
% samplingrate --> Sampling rate of the data
% matfilename --> name of the mat file to save the feature

f_all = [20 60];
f_gamma = [25 50];
window = 2;
noverlap = window - 1;
smoothfact = 15;

lfpstruc.data = lfp;
lfpstruc.timestamps = timestamps;
lfpstruc.samplingRate = samplingrate;
%Calculate PSS
[specslope,spec] = bz_PowerSpectrumSlope(lfpstruc,window,window-noverlap,...
    'nfreqs',200,'frange',f_all,'IRASA',true);
t_gclu = specslope.timestamps;
specdt = 1./specslope.samplingRate;
gammaFFTspec = specslope.resid';
gammaFFTspec(gammaFFTspec<0)=0;

IRASAsmooth_b = spec.IRASAsmooth';
gammaFFTspec_raw = 10.^spec.amp';

% Remove transients before calculating SW histogram
zFFTspec = NormToInt(spec.amp,'modZ');
totz = NormToInt(abs(sum(zFFTspec,2)),'modZ');
badtimes_G = find(totz>3);

gammaFFTfreqs = specslope.freqs';
gammafreqs = (gammaFFTfreqs>=f_gamma(1) & gammaFFTfreqs<=f_gamma(2));
gratio = max((gammaFFTspec(gammafreqs,:)),[],1);

gratio(badtimes_G) = nan;     %Remove transients
gratio = smooth(gratio,smoothfact./specdt);

% Remove ignoretimes (after smoothing), before normalizoing
if ~isempty(badtimes_G) 
	ignoretimeIDX = isnan(gratio);
    gratio(ignoretimeIDX) = [];
    t_gclu(ignoretimeIDX) = [];
end

gratio = bz_NormToRange(gratio,[0 1]);

GammaBand.data = gratio;
GammaBand.spec = gammaFFTspec;
GammaBand.timestamps = t_gclu;
GammaBand.freqs = gammaFFTfreqs;

save(matfilename,'GammaBand');


