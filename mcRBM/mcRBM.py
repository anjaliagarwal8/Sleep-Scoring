import sys
from scipy.io import loadmat, savemat
from ConfigParser import *
import numpy as np
import torch

######################################################
# mcRBM trainer: sweeps over the training set.
# For each batch of samples compute derivatives to update the parameters
# at the training samples and at the negative samples drawn calling HMC sampler.
def train_mcRBM():
    
    config = ConfigParser()
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
    totnumcases = d["whitendata"].shape[0]
    d = d["whitendata"][0:floor(totnumcases/batch_size)*batch_size,:].copy() 
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
    
    
    # start training
    for epoch in range(num_epochs):

        print("Epoch " + str(epoch + 1))
        
        # anneal learning rates
        epsilonVFc    = epsilonVF/max(1,epoch/20)
        epsilonFHc    = epsilonFH/max(1,epoch/20)
        epsilonbc    = epsilonb/max(1,epoch/20)
        epsilonw_meanc = epsilonw_mean/max(1,epoch/20)
        epsilonb_meanc = epsilonb_mean/max(1,epoch/20)
        weightcost = weightcost_final
        
        if epoch <= startFH:
            epsilonFHc = 0 
        if epoch <= startwd:	
            weightcost = 0

        for batch in range(num_batches):

            # get current minibatch
            data = dev_dat.narrow(1,batch*batch_size,(batch + 1)*batch_size) # DxP (nr dims x nr samples)
            
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
            torch.sum(feat_mean, 1, out = bias_meaninc)
            
            # HMC sampling: draw an approximate sample from the model
            if doPCD == 0: # CD-1 (set negative data to current training samples)
                hmc_step, hmc_ave_rej = draw_HMC_samples(data,negdata,normdata,vel,gradient,normgradient,new_energy,old_energy,VF,FH,bias_cov,bias_vis,w_mean,bias_mean,hmc_step,hmc_step_nr,hmc_ave_rej,hmc_target_ave_rej,t1,t2,t3,t4,t5,t6,t7,thresh,feat,featsq,batch_size,feat_mean,length,lengthsq,normcoeff,small,num_vis)
            else: # PCD-1 (use previous negative data as starting point for chain)
                negdataini = negdata
                hmc_step, hmc_ave_rej = draw_HMC_samples(negdataini,negdata,normdata,vel,gradient,normgradient,new_energy,old_energy,VF,FH,bias_cov,bias_vis,w_mean,bias_mean,hmc_step,hmc_step_nr,hmc_ave_rej,hmc_target_ave_rej,t1,t2,t3,t4,t5,t6,t7,thresh,feat,featsq,batch_size,feat_mean,length,lengthsq,normcoeff,small,num_vis)
                
            # compute derivatives at the negative samples
            # normalize input data
            torch.mul(negdata, negdata, out = t6) # DxP
            torch.sum(t6, 0, out = lengthsq) # 1xP
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
            torch.sub(FHinc, torch.matmul(featsq, t2.T), out = FHinc) # HxO #check
            FHinc.mult(0.5)
            cmt.dot(FH,t2, target = t3) # HxP
            t3.mult(feat)
            VFinc.subtract_dot(normdata, t3.T) # VxH
            bias_covinc.add_sums(t2, axis = 1)
        
if __name__ == "__main__":
    
    train_mcRBM()
    