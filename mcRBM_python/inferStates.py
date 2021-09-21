"""
    THIS CODE IS UNDER THE BSD 2-Clause LICENSE. YOU CAN FIND THE COMPLETE
                        FILE AT THE SOURCE DIRECTORY.

    Copyright (C) 2017 V.-M. Katsageorgiou - All rights reserved

    @author : vasiliki.katsageorgiou@gmail.com


                        Publication:
    A Novel Unsupervised Analysis of Electrophysiological
        Signals Reveals New Sleep Sub-stages in Mice


*****************************************************************************


Script written for inferring the latent states from a trained mcRBM model.


Run the script from a terminal as:
        python2 inferStates.py -f "path_to_experiment" -done "True/False" -m "saved_model"

where
    - "path_to_experiment" is a string with the path to the folder of
       the experiment you want to analyze.
    - "True/False" is Flag:
                    **Set as "True" if experiment is complete
                    **Set as "False" if experiment is on going and you
                      wish to analyze a saved model
    - "saved_model" is the name of the FINAL model. In case you want to
      analyze a stored model while training is on going, this string is
      not taken into account. You will be asked in the terminal to define
      the training epoch of the stored model you want to load.


OUTPUT: a folder named as "analysis" will be created in the experiment
        folder. Matrix with the latent activation will be in a sub-folder
        of the training epoch you are analyzing together with a matrix
        containing the unique binary latent states and a matrix with the
        data sample (epoch) id with its corresponding unique binary
        latent state.


<vkatsageorgiou@vassia-PC>
"""


import os
import sys
import numpy as np
from numpy.random import RandomState
#import cudamat as cmt
from scipy.io import loadmat, savemat
#import gpu_lock # put here you locking system package, if any
import configparser
import PIL.Image



class GetStates(object):
    '''
    Object containing all the functions needed to infer the latent
    activations of a trained mcRBM model.
    '''

    def __init__(self):
        # directory containing all the configuration files for the experiment
#        self.refDir = refDir
#        # id of the epoch to be analysed
#        #self.epochID = int(epochID)
#        self.expDoneFlag = expDoneFlag
#        self.model = finalModel
#
        np.random.seed(124)
#        self.prng = RandomState(123)
#
#        self.saveDir = self.refDir

    #-- Data Loading function:
    
    def loadData(self):
#        os.chdir(self.saveDir)
#        # Get current path:
#        print("Analysing experiment : ", os.getcwd())

        # load visData.npz file:
        visData = loadmat('visData.mat')
#        dataFile = np.load(visData)
        self.d = visData['visData']
        self.obsKeys = visData['obsKeys'].astype(int)

    #-- Infer Latent Activation Part:
    def computeStates(self):
        '''
        Method for inferring the latent states & analyse them.
        '''
        p_hc, p_hm = self.hidden_activation()
        self.p_all = np.concatenate((p_hc, p_hm), axis=1)

#        if not os.path.isdir('analysis'):
#            os.makedirs('analysis')
#        os.chdir('analysis')
#
#        if not os.path.isdir('epoch%d' %self.epochID):
#            os.makedirs('epoch%d' %self.epochID)
#
#        os.chdir('epoch%d' %self.epochID)
#        # Get current path:
#        print("Storing in...", os.getcwd())

        #savemat('lStatesProb_%i.mat' %self.epochID, mdict={'p_all':self.p_all})

#        if not os.path.isdir('hcActivation'):
#            os.makedirs('hcActivation')
#        if not os.path.isdir('hmActivation'):
#            os.makedirs('hmActivation')

        image = PIL.Image.fromarray(np.uint8(p_hc*255.))
        image.save('c%i.png' % self.epochID)

        image = PIL.Image.fromarray(np.uint8(p_hm*255.))
        image.save('m%i.png' % self.epochID)


        # Binarize the latent activations :
        self.binary_latentActivation = (self.p_all >= 0.5).astype(int)
        np.savez_compressed('latentStates.npz', probabilities=self.p_all, binary=self.binary_latentActivation)
        #savemat('latentStates.mat',{probabilities:self.p_all, binary:self.binary_latentActivation})
        
    # Function that computes the unique binary latent states
    def computeUniqueStates(self):
        '''
        Method for computing and storing the unique latent states.
        '''
        # Compute the Unique Binary Latent Activations :
        uniqueAct, p_unique = self.compute_uniques(self.binary_latentActivation, self.p_all)

        del self.p_all

        # Check if there are hidden_units that are always off :
        print("Checking if there are hidden_units that are always off..")
        print("The sum of the unique latent activations' columns is : ", np.sum(uniqueAct, axis=0))

        with open ('latentStatesInfo.txt','w') as f:
            f.write("\n The number of the unique latent activations is : %s" %uniqueAct.shape[0])
            f.write("\n The sum of the unique latent activations' columns is : %s" %np.sum(uniqueAct, axis=0))
            f.close()

        # Create a new array with the unique latent_states:
        # 1st Column = unique_state ID
        # 2nd Column = unique_state population
        # the binary latent representation follows in the next columns
        uniqueAct2 = np.insert(uniqueAct, 0, 0, axis=1)
        uniqueAct2 = np.insert(uniqueAct2, 0, 0, axis=1)

        # We do the same also to the obsKeys array, so that we know later
        # at each frame which latent state appears.
        #FrameIDs = FrameIDs.reshape(-1,1)
        self.obsKeys = np.insert(self.obsKeys, 1, 0, axis=1)
        # iterate through the unique latent states
        for i in range(uniqueAct.shape[0]):
            temp_idx = np.where(np.all(self.binary_latentActivation==uniqueAct[i,:], axis=1))[0]

            uniqueAct2[i, 0] = i
            uniqueAct2[i, 1] = len(temp_idx)

            self.obsKeys[temp_idx, 1] = i

        #savemat('uniqueStates.mat', mdict={'uniqueStates':uniqueAct2})
        #savemat('obsKeys.mat', mdict={'obsKeys':self.obsKeys})
        np.savez_compressed('uniqueStates.npz', uniqueStates=uniqueAct2, probabilities=p_unique)
        np.savez('obsKeys.npz', obsKeys=self.obsKeys)

    # Function that computes the unique binary latent states
    def compute_uniques(self, p_h_bin, p_h):
        '''
        Method for computing the unique latent states.
        '''

        # Compute the Unique Binary Latent Activations :
        tmpUnique = np.unique(p_h_bin.view(np.dtype((np.void, p_h_bin.dtype.itemsize*p_h_bin.shape[1]))), return_index = True, return_counts = True)
        uniqueAct = tmpUnique[0].view(p_h_bin.dtype).reshape(-1, p_h_bin.shape[1])
        uniqueFramesID = tmpUnique[1]
        uniqueFramesID = uniqueFramesID.reshape(-1,1)
        uniqueCount = tmpUnique[2]
        p_unique = p_h[uniqueFramesID[:,0],:]

        print( "The number of the unique latent activations is :", uniqueAct.shape[0])

        return uniqueAct, p_unique #, uniqueFrames

    # The logistic function
    def logisticFunc(self, x):
        '''
        Logistic function
        '''
        return 1./(1.+np.exp(-x))

    # Function that computes the latent activation of a trained mcRBM
    def hidden_activation(self):
        '''
        Function that infers the latent states.
        '''
        ws_temp = loadmat('ws_temp.mat')
#        if self.expDoneFlag=='True':
#            ws_temp = loadmat(self.model)
#        else:
#            temp_model = raw_input("Please enter the training epoch you want to analyse: ")
#            ws_temp = loadmat('./weights/ws_temp%d.mat' %int(temp_model))
        w_mean = ws_temp['w_mean']
        FH = ws_temp['FH']
        VF = ws_temp['VF']
        bias_cov = ws_temp['bias_cov']
        bias_mean = ws_temp['bias_mean']
        self.epochID = ws_temp['epoch']

        # Compute the probabilities of the covariance units:
        # FH has the opposite sign w.r.t. the paper, hence here it must be flipped

        # normalise data for covariance hidden:
        dsq = np.square(self.d)
        lsq = np.sum(dsq, axis=0)
        lsq /= self.d.shape[1]
        lsq += np.spacing(1)
        l = np.sqrt(lsq)
        normD = self.d/l

        # compute logistic_covarinace_argument:
        logisticArg_c = (-0.5*np.dot(FH.T, np.square(np.dot(VF.T, normD.T))) + bias_cov).T

        # Check what is going on:
        print( "logisticArg_c min : ", np.min(logisticArg_c))
        print( "logisticArg_c max : ", np.max(logisticArg_c))
        print( "logisticArg_c mean : ", np.mean(logisticArg_c))
        print( np.isfinite(logisticArg_c).all())

        # compute hidden_covariance probabilities:
        p_hc = self.logisticFunc(logisticArg_c)

        # compute logistic_mean_argument:
        # use unnormalised data for mean hidden
        logisticArg_m = np.dot(self.d, w_mean) + bias_mean.T

        # Check what is going on:
        print( "logisticArg_m min : ", np.min(logisticArg_m))
        print( "logisticArg_m max : ", np.max(logisticArg_m))
        print( "logisticArg_m mean : ", np.mean(logisticArg_m))
        print( np.isfinite(logisticArg_m).all())

        # compute hidden_mean probabilities:
        p_hm = self.logisticFunc(logisticArg_m)

        return p_hc, p_hm



if __name__ == "__main__":

#    import argparse
#
#    parser = argparse.ArgumentParser()
#    parser.add_argument('-f', help='Experiment path', default=False)
#    parser.add_argument('-done', help='Experiment done flag', default=False)
#    parser.add_argument('-m', help='Saved model name')
#    args = parser.parse_args()

    print( 'Initialization...')
    model = GetStates()

    print( 'Loading data...')
    model.loadData()

    print( 'Computing latent states...')
    model.computeStates()

    print( 'Computing the unique binary latent states...')
    model.computeUniqueStates()
