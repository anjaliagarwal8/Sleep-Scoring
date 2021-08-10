import numpy as np
import torch




######################################################
# mcRBM trainer: sweeps over the training set.
# For each batch of samples compute derivatives to update the parameters
# at the training samples and at the negative samples drawn calling HMC sampler.
def train_mcRBM():
    
    num_epochs = 
    batch_size = 
    startFH = 
    startwd = 
    doPCD = 
    
    # model parameters
    num_fac = 
    num_hid_cov =  
    num_hid_mean =  
    apply_mask =  
    
    # load data
    data_file_name =  
    d = loadmat(data_file_name) # input in the format PxD (P vectorized samples with D dimensions)
    totnumcases = d["whitendata"].shape[0]
    d = d["whitendata"][0:floor(totnumcases/batch_size)*batch_size,:].copy() 
    totnumcases = d.shape[0]
    num_vis =  d.shape[1]
    num_batches = int(totnumcases/batch_size)
    dev_dat = torch.tensor(d.T) # VxP 
    
    # training parameters
    epsilon = 
    epsilonVF = 2*epsilon
    epsilonFH = 0.02*epsilon
    epsilonb = 0.02*epsilon
    epsilonw_mean = 0.2*epsilon
    epsilonb_mean = 0.1*epsilon
    weightcost_final =  

    # HMC setting
    hmc_step_nr = 
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

if __name__ == "__main__":
    
    train_mcRBM()
    