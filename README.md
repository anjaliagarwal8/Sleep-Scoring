# Automated Sleep Scoring



For running the MATLAB code for sleep scoring follow the following steps:

1. Download the Sleep Scoring code from github https://github.com/anjaliagarwal8/Sleep-Scoring.git
2. Proceed to the folder mcRBM_matlab/
3. Add the data you want to score in the same folder named as 'data.mat'. 
4. Edit the file input_configuration.mat to modify parameters involved in the mcRBM algorithm, number of hidden units, learning rate, number of epochs, etc.
5. Run the main.m file in Matlab. 
6. The variables needed for finding the number of latent states of sleep will be saved as variables.mat after the program has successfully run for the provided number of epochs. 
7. After training the model, to find the unique latent states present in the sleep dataset, run the inferStates.m file. The corresponding results will be saved in mat files along with the necessary plots. 
8. Run analyzeStates.m file to get the BoxPlots of each latent state with respect to each feature present in the dataset. The figures will be saved in BoxPlots folder for each latent state.
9. To get the distribution of latent states among the three stages, WAKE, NREM and REM run the stageDistribution.m file. It will plot the distribution in the form of a heatmap and will save the distribution in the mat file.
10. To get the histogram for the latent states, run the statesHistogram.m file.
11. To compute the transition probability matrix, run the ComputeTransitions.m file. This program computes the transition probability matrix along with the directed graph of the unique latent states. The threshold for the occurrence of latent states can be set in this file. The default value is set as 1, i.e. any latent state occurring only once in the given dataset will be removed.


#### References
- Watson, B. O., & Buzsáki, G. (2015). Sleep, Memory & Brain Rhythms. Daedalus, 144(1), 67–82. https://doi.org/10.1162/DAED_a_00318
- Watson, B. O., Levenstein, D., Greene, J. P., Gelinas, J. N., & Buzsáki, G. (2016). Network Homeostasis and State Dynamics of Neocortical Sleep. Neuron, 90(4), 839–852. https://doi.org/10.1016/j.neuron.2016.03.036
- Katsageorgiou VM, Sona D, Zanotto M, Lassi G, Garcia-Garcia C, et al. (2018) A novel unsupervised analysis of electrophysiological signals reveals new sleep substages in mice. PLOS Biology 16(5): e2003663. https://doi.org/10.1371/journal.pbio.2003663
- Ranzato, M., & Hinton, G.E. (2010). Modeling pixel means and covariances using factorized third-order boltzmann machines. 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2551-2558.

