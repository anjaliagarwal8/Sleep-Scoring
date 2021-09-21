import sys
from scipy.io import loadmat, savemat
import configparser 
import numpy as np
import torch
import pylab
import matplotlib.pyplot as plt
from numpy.random import RandomState

######################################################################
# compute the value of the free energy at a given input
# F = - sum log(1+exp(- .5 FH (VF data/norm(data))^2 + bias_cov)) +...
#     - sum log(1+exp(w_mean data + bias_mean)) + ...
#     - bias_vis data + 0.5 data^2
# NOTE: FH is constrained to be positive 
# (in the paper the sign is negative but the sign in front of it is also flipped)
def compute_energy_mcRBM(data,normdata,vel,energy,VF,FH,bias_cov,bias_vis,w_mean,bias_mean,t1,t2,t6,feat,featsq,feat_mean,length,lengthsq,normcoeff,small,num_vis,store):
    # normalize input data vectors
    torch.mul(data, data, out = t6)
    torch.sum(t6, 0, keepdims = True,out = lengthsq)
    torch.mul(lengthsq, 0.5, out = energy)
    torch.mul(lengthsq, 1./num_vis, out = lengthsq)
    torch.add(lengthsq, small, out = lengthsq)
    torch.sqrt(lengthsq, out = length)
    torch.reciprocal(length, out = normcoeff)
    torch.mul(data, normcoeff, out = normdata)
    ## potential
    # covariance contribution
    torch.matmul(VF.T, normdata, out = feat)
    torch.mul(feat, feat, out = featsq)
    torch.matmul(FH.T, featsq, out = t1)
    torch.mul(t1, -0.5, out = t1)
    torch.add(t1, bias_cov, out = t1)
    torch.exp(t1, out = t1)
    torch.add(t1, 1, out = t2)
    torch.log(t2, out = t2)
    torch.mul(t2, -1, out = t2)
    torch.add(energy, torch.sum(t2, 0,keepdims = True), out = energy)
    # mean contribution
    torch.matmul(w_mean.T, data, out = feat_mean)
    torch.add(feat_mean, bias_mean, out = feat_mean)
    torch.exp(feat_mean, out = feat_mean)
    torch.add(feat_mean, 1, out = feat_mean)
    torch.log(feat_mean, out = feat_mean)
    torch.mul(feat_mean, -1, out = feat_mean)
    torch.add(energy, torch.sum(feat_mean, 0,keepdims = True), out = energy)
    # visible bias term
    torch.mul(data, bias_vis, out = t6)
    torch.mul(t6, -1, out = t6)
    torch.add(energy, torch.sum(t6, 0,keepdims = True), out = energy)
    if store == False:
        # kinetic
        torch.mul(vel, vel, out = t6)
    else:
        # kinetic
        torch.mul(data,data, out = t6)
        
    torch.add(energy, torch.mul(torch.sum(t6, 0, keepdims = True), 0.5), out = energy)
    
    
#################################################################
# compute the derivative if the free energy at a given input
def compute_gradient_mcRBM(data,normdata,VF,FH,bias_cov,bias_vis,w_mean,bias_mean,t1,t2,t3,t4,t6,feat,featsq,feat_mean,gradient,normgradient,length,lengthsq,normcoeff,small,num_vis):
    # normalize input data
    torch.mul(data, data, out = t6) # DxP
    torch.sum(t6, 0, keepdims = True, out = lengthsq) # 1xP
    torch.mul(lengthsq, 1./num_vis, out = lengthsq) # normalize by number of components (like std)
    torch.add(lengthsq, small, out = lengthsq) 
    torch.sqrt(lengthsq, out = length)
    torch.reciprocal(length, out = normcoeff) # 1xP
    torch.mul(data, normcoeff, out = normdata)
    
    torch.matmul(VF.T, normdata, out = feat)
    torch.mul(feat, feat, out = featsq)
    torch.matmul(FH.T, featsq, out = t1)
    torch.mul(t1, -0.5, out = t1)
    torch.add(t1, bias_cov, out = t1)
    torch.sigmoid(t1, out = t2)
    torch.matmul(FH, t2, out = t3)
    torch.mul(t3, feat, out = t3)
    torch.matmul(VF, t3, out = normgradient)
    # final bprop through normalization
    torch.mul(length, lengthsq, out = normcoeff)
    torch.reciprocal(normcoeff, out = normcoeff)
    torch.mul(normgradient, data, out = gradient)
    torch.sum(gradient, 0, keepdims = True,out = t4)
    torch.mul(t4, -1./num_vis, out = t4)
    torch.mul(data, t4, out = gradient)
    torch.mul(normgradient, lengthsq, out = t6)
    torch.add(gradient, t6, out = gradient)
    torch.mul(gradient, normcoeff, out = gradient)
    # add quadratic term gradient
    torch.add(gradient, data, out = gradient)
    # add visible bias term
    torch.add(gradient, torch.mul(bias_vis, -1), out = gradient)
    # add MEAN contribution to gradient
    torch.matmul(w_mean.T, data, out = feat_mean)
    torch.add(feat_mean, bias_mean, out = feat_mean)
    torch.sigmoid(feat_mean, out = feat_mean)
    torch.sub(gradient, torch.matmul(w_mean, feat_mean), out = gradient)
    
    
############################################################3
# Hybrid Monte Carlo sampler
def draw_HMC_samples(data,negdata,normdata,vel,gradient,normgradient,new_energy,old_energy,VF,FH,bias_cov,bias_vis,w_mean,bias_mean,hmc_step,hmc_step_nr,hmc_ave_rej,hmc_target_ave_rej,t1,t2,t3,t4,t5,t6,t7,thresh,feat,featsq,batch_size,feat_mean,length,lengthsq,normcoeff,small,num_vis):
    torch.randn(vel.size(),out = vel)
    negdata = data.clone().detach()
    compute_energy_mcRBM(negdata,normdata,vel,old_energy,VF,FH,bias_cov,bias_vis,w_mean,bias_mean,t1,t2,t6,feat,featsq,feat_mean,length,lengthsq,normcoeff,small,num_vis,False)
    compute_gradient_mcRBM(negdata,normdata,VF,FH,bias_cov,bias_vis,w_mean,bias_mean,t1,t2,t3,t4,t6,feat,featsq,feat_mean,gradient,normgradient,length,lengthsq,normcoeff,small,num_vis)
    # half step
    torch.add(vel,torch.mul(gradient, -0.5*hmc_step), out = vel)
    torch.add(negdata, torch.mul(vel,hmc_step), out = negdata)
    # full leap-frog steps
    for ss in range(hmc_step_nr - 1):
        ## re-evaluate the gradient
        compute_gradient_mcRBM(negdata,normdata,VF,FH,bias_cov,bias_vis,w_mean,bias_mean,t1,t2,t3,t4,t6,feat,featsq,feat_mean,gradient,normgradient,length,lengthsq,normcoeff,small,num_vis)
        # update variables
        torch.add(vel, torch.mul(gradient, -hmc_step), out = vel)
        torch.add(negdata, torch.mul(vel,hmc_step), out = negdata)
    
    # final half-step
    compute_gradient_mcRBM(negdata,normdata,VF,FH,bias_cov,bias_vis,w_mean,bias_mean,t1,t2,t3,t4,t6,feat,featsq,feat_mean,gradient,normgradient,length,lengthsq,normcoeff,small,num_vis)
    torch.add(vel, torch.mul(gradient, -0.5*hmc_step), out = vel)
    # compute new energy
    compute_energy_mcRBM(negdata,normdata,vel,new_energy,VF,FH,bias_cov,bias_vis,w_mean,bias_mean,t1,t2,t6,feat,featsq,feat_mean,length,lengthsq,normcoeff,small,num_vis,False)
    # rejecton
    torch.sub(old_energy, new_energy, out = thresh)
    torch.exp(thresh, out = thresh)
    t4.random_()
    torch.mul(torch.le(t4,thresh), 1., out = t4)
    #    update negdata and rejection rate
    torch.mul(t4, -1, out = t4)
    torch.add(t4, 1, out = t4) # now 1's detect rejections
    torch.sum(t4, 1, keepdims = True,out = t5)
    #t5.cpu().data.numpy()
    rej = t5[0,0]/batch_size
    torch.mul(data, t4, out = t6)
    torch.mul(negdata, t4, out = t7)
    torch.sub(negdata, t7, out = negdata)
    torch.add(negdata, t6, out = negdata)
    hmc_ave_rej = 0.9*hmc_ave_rej + 0.1*rej
    if hmc_ave_rej < hmc_target_ave_rej:
        hmc_step = min(hmc_step*1.01,0.25)
    else:
        hmc_step = max(hmc_step*0.99,.001)
    return hmc_step, hmc_ave_rej
 
    
######################################################
# mcRBM trainer: sweeps over the training set.
# For each batch of samples compute derivatives to update the parameters
# at the training samples and at the negative samples drawn calling HMC sampler.
def train_mcRBM():
    
    np.random.seed(124)
    prng =  RandomState(123)
        
    config = configparser.ConfigParser()
    config.read('input_configuration')

    verbose = config.getint('VERBOSITY','verbose')

    num_epochs = config.getint('MAIN_PARAMETER_SETTING','num_epochs')
    batch_size = config.getint('MAIN_PARAMETER_SETTING','batch_size')
    startFH = config.getint('MAIN_PARAMETER_SETTING','startFH')
    startwd = config.getint('MAIN_PARAMETER_SETTING','startwd')
    doPCD = config.getint('MAIN_PARAMETER_SETTING','doPCD')
    
    # model parameters
    num_fac = config.getint('MODEL_PARAMETER_SETTING','num_fac')
    num_hid_cov =  config.getint('MODEL_PARAMETER_SETTING','num_hid_cov')
    num_hid_mean =  config.getint('MODEL_PARAMETER_SETTING','num_hid_mean')
    apply_mask =  config.getint('MODEL_PARAMETER_SETTING','apply_mask')
    
    # load data
    data_file_name =  config.get('DATA','data_file_name')
    d = loadmat(data_file_name) # input in the format PxD (P vectorized samples with D dimensions)
    d = d['d']
    d = d.astype(np.float32)
    
    # Scaling the data
    dMinRow = np.min(d, axis = 0)
    dMaxRow = np.max(d, axis = 0)
    d = 10.*((d - dMinRow) / (dMaxRow - dMinRow) - 0.5)
    
    permIdx = prng.permutation(d.shape[0])

    #d = d[permIdx,:]
    
    #totnumcases = d["whitendata"].shape[0]
    #d = d["whitendata"][0:int(pylab.floor(totnumcases/batch_size))*batch_size,:].copy() 
    totnumcases = d.shape[0]
    num_vis =  d.shape[1]
    num_batches = int(totnumcases/batch_size)
    dev_dat = torch.tensor(d.T) # VxP 
    
    # training parameters
    epsilon = config.getfloat('OPTIMIZER_PARAMETERS','epsilon')
    epsilonVF = 2*epsilon
    epsilonFH = 0.02*epsilon
    epsilonb = 0.02*epsilon
    epsilonw_mean = 0.2*epsilon
    epsilonb_mean = 0.1*epsilon
    weightcost_final =  config.getfloat('OPTIMIZER_PARAMETERS','weightcost_final')

    # HMC setting
    hmc_step_nr = config.getint('HMC_PARAMETERS','hmc_step_nr')
    hmc_step =  0.01
    hmc_target_ave_rej =  config.getfloat('HMC_PARAMETERS','hmc_target_ave_rej')
    hmc_ave_rej =  hmc_target_ave_rej
    
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
        
    # initialize weights
    VF = torch.tensor(np.array(0.02 * np.random.randn(num_vis, num_fac), dtype=np.float32, order='F')) # VxH
    if apply_mask == 0:
        FH = torch.tensor( np.array( np.eye(num_fac,num_hid_cov), dtype=np.float32, order='F')  ) # HxO
    else:
        dd = loadmat('your_FHinit_mask_file.mat') # see CVPR2010paper_material/topo2D_3x3_stride2_576filt.mat for an example
        FH = torch.tensor( np.array( dd["FH"], dtype=np.float32, order='F')  )
    bias_cov = torch.tensor( np.array(2.0*np.ones((num_hid_cov, 1)), dtype=np.float32, order='F') )
    bias_vis = torch.tensor( np.array(np.zeros((num_vis, 1)), dtype=np.float32, order='F') )
    w_mean = torch.tensor( np.array( 0.05 * np.random.randn(num_vis, num_hid_mean), dtype=np.float32, order='F') ) # VxH
    bias_mean = torch.tensor( np.array( -2.0*np.ones((num_hid_mean,1)), dtype=np.float32, order='F') )
    
    savemat("variables_init.mat",{'VF':VF.cpu().data.numpy(),'FH':FH.cpu().data.numpy(),'bias_cov': bias_cov.cpu().data.numpy(), 'bias_vis': bias_vis.cpu().data.numpy(), 'w_mean': w_mean.cpu().data.numpy(), 'bias_mean': bias_mean.cpu().data.numpy()})
    
    # initialize variables to store derivatives 
    VFinc = torch.tensor( np.array(np.zeros((num_vis, num_fac)), dtype=np.float32, order='F'))
    FHinc = torch.tensor( np.array(np.zeros((num_fac, num_hid_cov)), dtype=np.float32, order='F'))
    bias_covinc = torch.tensor( np.array(np.zeros((num_hid_cov, 1)), dtype=np.float32, order='F'))
    bias_visinc = torch.tensor( np.array(np.zeros((num_vis, 1)), dtype=np.float32, order='F'))
    w_meaninc = torch.tensor( np.array(np.zeros((num_vis, num_hid_mean)), dtype=np.float32, order='F'))
    bias_meaninc = torch.tensor( np.array(np.zeros((num_hid_mean, 1)), dtype=np.float32, order='F'))

    # initialize temporary storage
    data = torch.tensor( np.array(np.empty((num_vis, batch_size)), dtype=np.float32, order='F')) # VxP
    normdata = torch.tensor( np.array(np.empty((num_vis, batch_size)), dtype=np.float32, order='F')) # VxP
    negdataini = torch.tensor( np.array(np.empty((num_vis, batch_size)), dtype=np.float32, order='F')) # VxP
    feat = torch.tensor( np.array(np.empty((num_fac, batch_size)), dtype=np.float32, order='F'))
    featsq = torch.tensor( np.array(np.empty((num_fac, batch_size)), dtype=np.float32, order='F'))
    negdata = torch.tensor( np.array(np.random.randn(num_vis, batch_size), dtype=np.float32, order='F'))
    old_energy = torch.tensor( np.array(np.zeros((1, batch_size)), dtype=np.float32, order='F'))
    new_energy = torch.tensor( np.array(np.zeros((1, batch_size)), dtype=np.float32, order='F'))
    energy = torch.tensor( np.array(np.zeros((1, batch_size)), dtype=np.float32, order='F'))
    gradient = torch.tensor( np.array(np.empty((num_vis, batch_size)), dtype=np.float32, order='F')) # VxP
    normgradient = torch.tensor( np.array(np.empty((num_vis, batch_size)), dtype=np.float32, order='F')) # VxP
    thresh = torch.tensor( np.array(np.zeros((1, batch_size)), dtype=np.float32, order='F'))
    feat_mean = torch.tensor( np.array(np.empty((num_hid_mean, batch_size)), dtype=np.float32, order='F'))
    vel = torch.tensor( np.array(np.random.randn(num_vis, batch_size), dtype=np.float32, order='F'))
    length = torch.tensor( np.array(np.zeros((1, batch_size)), dtype=np.float32, order='F')) # 1xP
    lengthsq = torch.tensor( np.array(np.zeros((1, batch_size)), dtype=np.float32, order='F')) # 1xP
    normcoeff = torch.tensor( np.array(np.zeros((1, batch_size)), dtype=np.float32, order='F')) # 1xP
    if apply_mask==1: # this used to constrain very large FH matrices only allowing to change values in a neighborhood
        dd = loadmat('your_FHinit_mask_file.mat') 
        mask = torch.tensor( np.array(dd["mask"], dtype=np.float32, order='F'))
    normVF = 1    
    small = 0.5
    
    # other temporary vars
    t1 = torch.tensor( np.array(np.empty((num_hid_cov, batch_size)), dtype=np.float32, order='F'))
    t2 = torch.tensor( np.array(np.empty((num_hid_cov, batch_size)), dtype=np.float32, order='F'))
    t3 = torch.tensor( np.array(np.empty((num_fac, batch_size)), dtype=np.float32, order='F'))
    t4 = torch.tensor( np.array(np.empty((1,batch_size)), dtype=np.float32, order='F'))
    t5 = torch.tensor( np.array(np.empty((1,1)), dtype=np.float32, order='F'))
    t6 = torch.tensor( np.array(np.empty((num_vis, batch_size)), dtype=np.float32, order='F'))
    t7 = torch.tensor( np.array(np.empty((num_vis, batch_size)), dtype=np.float32, order='F'))
    t8 = torch.tensor( np.array(np.empty((num_vis, num_fac)), dtype=np.float32, order='F'))
    t9 = torch.tensor( np.array(np.zeros((num_fac, num_hid_cov)), dtype=np.float32, order='F'))
    t10 = torch.tensor( np.array(np.empty((1,num_fac)), dtype=np.float32, order='F'))
    t11 = torch.tensor( np.array(np.empty((1,num_hid_cov)), dtype=np.float32, order='F'))
    
    meanEnergy = np.zeros(num_epochs)
    minEnergy = np.zeros(num_epochs)
    maxEnergy = np.zeros(num_epochs)
    
    
    # start training
    for epoch in range(50):

        print("Epoch " + str(epoch + 1))
        
        # anneal learning rates
#        epsilonVFc    = epsilonVF/max(1,epoch/20)
#        epsilonFHc    = epsilonFH/max(1,epoch/20)
#        epsilonbc    = epsilonb/max(1,epoch/20)
#        epsilonw_meanc = epsilonw_mean/max(1,epoch/20)
#        epsilonb_meanc = epsilonb_mean/max(1,epoch/20)
#        weightcost = weightcost_final
        
        # no annealing is used because learning
        # was stopping too early
        epsilonVFc = epsilonVF
        epsilonFHc = epsilonFH
        epsilonbc = epsilonb
        epsilonw_meanc = epsilonw_mean
        epsilonb_meanc = epsilonb_mean
        weightcost = weightcost_final
        
        if epoch <= startFH:
            epsilonFHc = 0 
        if epoch <= startwd:    
            weightcost = 0

        for batch in range(num_batches):

            # get current minibatch
            data = dev_dat[:,batch*batch_size:(batch + 1)*batch_size] # DxP (nr dims x nr samples)
            
            # normalize input data
            torch.mul(data, data, out = t6) # DxP
            torch.sum(t6, 0, keepdims = True, out = lengthsq) # 1xP
            torch.mul(lengthsq, 1./num_vis, out = lengthsq) # normalize by number of components (like std)
            torch.add(lengthsq, small, out = lengthsq) # small avoids division by 0
            torch.sqrt(lengthsq, out = length)
            torch.reciprocal(length, out = normcoeff) # 1xP
            torch.mul(data, normcoeff, out = normdata) # normalized data
            
            ## compute positive sample derivatives
            # covariance part
            torch.matmul(VF.T, normdata, out = feat) # HxP (nr facs x nr samples)
            torch.mul(feat, feat, out = featsq)   # HxP
            torch.matmul(FH.T,featsq, out = t1) # OxP (nr cov hiddens x nr samples)
            torch.mul(t1,-0.5,out = t1)
            torch.add(t1,bias_cov, out = t1) # OxP
            torch.sigmoid(t1, out = t2) # OxP
            torch.matmul(featsq, t2.T, out = FHinc) # HxO
            torch.matmul(FH,t2, out = t3) # HxP
            torch.mul(t3,feat, out = t3)
            torch.matmul(normdata, t3.T, out = VFinc) # VxH
            torch.sum(t2, 1, keepdims= True, out = bias_covinc)
            torch.mul(bias_covinc, -1, out = bias_covinc)
            
            # visible bias
            torch.sum(data, 1, keepdims = True, out = bias_visinc)
            torch.mul(bias_visinc, -1, out = bias_visinc)
            # mean part
            torch.matmul(w_mean.T, data, out = feat_mean) # HxP (nr mean hiddens x nr samples)
            torch.add(feat_mean, bias_mean, out = feat_mean) # HxP
            torch.sigmoid(feat_mean, out = feat_mean) # HxP
            torch.mul(feat_mean, -1, out = feat_mean)
            torch.matmul(data, feat_mean.T, out = w_meaninc)
            torch.sum(feat_mean, 1, keepdims = True,out = bias_meaninc)
            
            # HMC sampling: draw an approximate sample from the model
            if doPCD == 0: # CD-1 (set negative data to current training samples)
                hmc_step, hmc_ave_rej = draw_HMC_samples(data,negdata,normdata,vel,gradient,normgradient,new_energy,old_energy,VF,FH,bias_cov,bias_vis,w_mean,bias_mean,hmc_step,hmc_step_nr,hmc_ave_rej,hmc_target_ave_rej,t1,t2,t3,t4,t5,t6,t7,thresh,feat,featsq,batch_size,feat_mean,length,lengthsq,normcoeff,small,num_vis)
            else: # PCD-1 (use previous negative data as starting point for chain)
                negdataini = torch.clone(negdata)
                hmc_step, hmc_ave_rej = draw_HMC_samples(negdataini,negdata,normdata,vel,gradient,normgradient,new_energy,old_energy,VF,FH,bias_cov,bias_vis,w_mean,bias_mean,hmc_step,hmc_step_nr,hmc_ave_rej,hmc_target_ave_rej,t1,t2,t3,t4,t5,t6,t7,thresh,feat,featsq,batch_size,feat_mean,length,lengthsq,normcoeff,small,num_vis)
             
            # compute derivatives at the negative samples
            # normalize input data
            torch.mul(negdata, negdata, out = t6) # DxP
            torch.sum(t6, 0, keepdims = True,out = lengthsq) # 1xP
            torch.mul(lengthsq, 1./num_vis, out = lengthsq) # normalize by number of components (like std)
            torch.add(lengthsq, small, out = lengthsq)
            torch.sqrt(lengthsq, out = length)
            torch.reciprocal(length, out = normcoeff) # 1xP
            torch.mul(negdata, normcoeff, out = normdata) # normalized data 
            
            # covariance part
            torch.matmul(VF.T, normdata, out = feat) # HxP 
            torch.mul(feat, feat, out = featsq)   # HxP
            torch.matmul(FH.T,featsq, out = t1) # OxP
            torch.mul(t1, -0.5, out = t1)
            torch.add(t1, bias_cov, out = t1) # OxP
            torch.sigmoid(t1, out = t2) # OxP
            torch.sub(FHinc, torch.matmul(featsq, t2.T), out = FHinc) # HxO 
            torch.mul(FHinc, 0.5, out = FHinc)
            torch.matmul(FH,t2, out = t3) # HxP
            torch.mul(t3, feat, out = t3)
            torch.sub(VFinc, torch.matmul(normdata, t3.T), out = VFinc) # VxH
            torch.add(bias_covinc, torch.sum(t2, 1,keepdims = True), out = bias_covinc)
            # visible bias
            torch.add(bias_visinc, torch.sum(negdata, 1,keepdims = True), out = bias_visinc)
            # mean part
            torch.matmul(w_mean.T, negdata, out = feat_mean) # HxP 
            torch.add(feat_mean, bias_mean, out = feat_mean) # HxP
            torch.sigmoid(feat_mean, out = feat_mean) # HxP
            torch.add(w_meaninc, torch.matmul(negdata, feat_mean.T), out = w_meaninc)
            torch.add(bias_meaninc, torch.sum(feat_mean, 1,keepdims = True), out = bias_meaninc)
            
            # update parameters
            torch.add(VFinc, torch.mul(torch.sign(VF), weightcost), out = VFinc) # L1 regularization
            torch.add(VF, torch.mul(VFinc, -epsilonVFc/batch_size), out = VF)
            # normalize columns of VF: normalize by running average of their norm 
            torch.mul(VF, VF, out = t8)
            torch.sum(t8, 0, keepdims = True,out = t10)
            torch.sqrt(t10)
            torch.sum(t10, 1, keepdims = True,out = t5)
            #t5 = t5.cpu().data.numpy()
            normVF = .95*normVF + (.05/num_fac) * t5[0,0] # estimate norm
            torch.reciprocal(t10, out = t10)
            torch.mul(VF, t10, out = VF) 
            torch.mul(VF, normVF, out = VF) 
            torch.add(bias_cov, torch.mul(bias_covinc, -epsilonbc/batch_size), out = bias_cov)
            torch.add(bias_vis, torch.mul(bias_visinc, -epsilonbc/batch_size), out = bias_vis)
            
            if epoch > startFH:
                
                torch.add(FHinc, torch.mul(torch.sign(FH), weightcost), out = FHinc) # L1 regularization
                torch.add(FH, torch.mul(FHinc, -epsilonFHc/batch_size), out = FH)
                # set to 0 negative entries in FH
                torch.mul(torch.ge(FH, 0), 1., out = t9)
                torch.mul(FH, t9, out = FH)
                if apply_mask==1:
                    torch.mul(FH, mask, out = FH)
    
                # normalize columns of FH: L1 norm set to 1 in each column
                torch.sum(FH, 0, keepdims = True,out = t11)
                torch.reciprocal(t11, out = t11)
                torch.mul(FH, t11, out = FH) 
               
            torch.add(w_meaninc, torch.mul(torch.sign(w_mean),weightcost), out = w_meaninc)
            torch.add(w_mean, torch.mul(w_meaninc, -epsilonw_meanc/batch_size), out = w_mean)
            torch.add(bias_mean, torch.mul(bias_meaninc, -epsilonb_meanc/batch_size), out = bias_mean)
            
#            if verbose == 1:
#                print("VF: " + '%3.2e' % VF.euclid_norm() + ", DVF: " + '%3.2e' % (VFinc.euclid_norm()*(epsilonVFc/batch_size)) + ", FH: " + '%3.2e' % FH.euclid_norm() + ", DFH: " + '%3.2e' % (FHinc.euclid_norm()*(epsilonFHc/batch_size)) + ", bias_cov: " + '%3.2e' % bias_cov.euclid_norm() + ", Dbias_cov: " + '%3.2e' % (bias_covinc.euclid_norm()*(epsilonbc/batch_size)) + ", bias_vis: " + '%3.2e' % bias_vis.euclid_norm() + ", Dbias_vis: " + '%3.2e' % (bias_visinc.euclid_norm()*(epsilonbc/batch_size)) + ", wm: " + '%3.2e' % w_mean.euclid_norm() + ", Dwm: " + '%3.2e' % (w_meaninc.euclid_norm()*(epsilonw_meanc/batch_size)) + ", bm: " + '%3.2e' % bias_mean.euclid_norm() + ", Dbm: " + '%3.2e' % (bias_meaninc.euclid_norm()*(epsilonb_meanc/batch_size)) + ", step: " + '%3.2e' % hmc_step  +  ", rej: " + '%3.2e' % hmc_ave_rej) 
#                sys.stdout.flush()
#            
            compute_energy_mcRBM(data,normdata,vel,energy,VF,FH,bias_cov,bias_vis,w_mean,bias_mean,t1,t2,t6,feat,featsq,feat_mean,length,lengthsq,normcoeff,small,num_vis,True)
#            energy.copy_to_host()
            meanEnergy[epoch] = np.mean(energy.cpu().data.numpy())
            minEnergy[epoch] = np.min(energy.cpu().data.numpy())
            maxEnergy[epoch] = np.max(energy.cpu().data.numpy())
            
            ax1.cla()
            ax1.plot(range(epoch), meanEnergy[0:epoch])
            ax1.plot(range(epoch), maxEnergy[0:epoch])
            ax1.plot(range(epoch), minEnergy[0:epoch])
            
        # back-up every once in a while 
        if np.mod(epoch,10) == 0:
#            VF.copy_to_host()
#            FH.copy_to_host()
#            bias_cov.copy_to_host()
#            w_mean.copy_to_host()
#            bias_mean.copy_to_host()
#            bias_vis.copy_to_host()
            savemat("ws_temp.mat", {'VF':VF.cpu().data.numpy(),'FH':FH.cpu().data.numpy(),'bias_cov': bias_cov.cpu().data.numpy(), 'bias_vis': bias_vis.cpu().data.numpy(),'w_mean': w_mean.cpu().data.numpy(), 'bias_mean': bias_mean.cpu().data.numpy(), 'epoch':epoch})    
    # final back-up
#    VF.copy_to_host()
#    FH.copy_to_host()
#    bias_cov.copy_to_host()
#    bias_vis.copy_to_host()
#    w_mean.copy_to_host()
    
#    bias_mean.copy_to_host()
    savemat("ws_fac" + str(num_fac) + "_cov" + str(num_hid_cov) + "_mean" + str(num_hid_mean)+".mat", {'VF':VF.cpu().data.numpy(),'FH':FH.cpu().data.numpy(),'bias_cov': bias_cov.cpu().data.numpy(), 'bias_vis': bias_vis.cpu().data.numpy(), 'w_mean': w_mean.cpu().data.numpy(), 'bias_mean': bias_mean.cpu().data.numpy(), 'epoch':epoch})
    savemat("training_energy_" + str(num_fac) + "_cov" + str(num_hid_cov) + "_mean" + str(num_hid_mean)+".mat", {'meanEnergy':meanEnergy, 'maxEnergy': maxEnergy, 'minEnergy': minEnergy, 'epoch':epoch})
    
 
if __name__ == "__main__":
    
    train_mcRBM()
    