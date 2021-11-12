# Automated Sleep Scoring



For running the MATLAB code for sleep scoring follow the following steps:

1. Download the Sleep Scoring code from github https://github.com/anjaliagarwal8/Sleep-Scoring.git
2. Proceed to the folder mcRBM_matlab/
3. Add the data you want to score in the same folder named as 'data.mat'. 
4. Edit the file input_configuration.mat to modify parameters involved in the mcRBM algorithm, number of hidden units, learning rate, number of epochs, etc.
5. Run the main.m file in Matlab. 
6. The variables needed for finding the number of latent states of sleep will be saved as variables.mat after the program has successfully run for the provided number of epochs. 
7. After training the model, to find the unique latent states present in the sleep dataset, run the inferStates.m file. The corresponding results will be saved in mat files along with the necessary plots. 
8. To get the distribution of latent states among the three stages, WAKE, NREM and REM run the stageDistribution.m file. It will plot the distribution in the form of a heatmap and will save the distribution in the mat file.
9. To get the histogram for the latent states, run the statesHistogram.m file.
10. To compute the transition probability matrix, run the ComputeTransitions.m file. This program computes the transition probability matrix along with the directed graph of the unique latent states. The threshold for the occurrence of latent states can be set in this file. The default value is set as 1, i.e. any latent state occurring only once in the given dataset will be removed.



The features used for Sleep State Clustering are extracted using Sleep Score Master code developed by Buzsaki Lab team. The features are then fed into Restricted Boltzmann Machine for making clusters of various sleep stages. 
