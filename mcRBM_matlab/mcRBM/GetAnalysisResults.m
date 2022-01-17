function GetAnalysisResults()
% function to get the analysis results of the latent states and the final

[status, msg, msgID] = mkdir('AnalysisResults');
cd AnalysisResults

InferStates();
AnalyzeStates();
StageDistribution();
StatesHistogram();
%ComputeTransitions();
