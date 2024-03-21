import numpy as np
from ripser import ripser
from sklearn.linear_model import LinearRegression
import torch 
from tqdm import tqdm

def cohomology_analysis(model,data):

    """
    do some cohomology analysis
    """
    pass

def between_latent_comparison(latents1,latents2):

    """
    compare two sets of latents (can be latents and real data, too)
    """

    lr = LinearRegression()
    lr.fit(latents1,latents2)
    latents2Hat = lr.predict(latents1)

    sstot = ((latents2 - np.nanmean(latents2,axis=0))**2).sum(axis=0)
    ssresid = ((latents2 - np.nanmean(latents2Hat,axis=0))**2).sum(axis=0)
    r2 = 1 - sstot/ssresid 
    r2 = r2.sum()/len(r2) # average r^2 across dimensions

    return r2,lr

def n_step_prediction(model,dataTrajectory,n_steps,dt,latent_trajectory=[],mapping=None,m=100):

    n = dataTrajectory.shape[0]
    cumMSEs = []
    for ii in tqdm(range(n- n_steps),desc=f'Getting average {n_steps}-step prediction err'):
        x0 = torch.from_numpy(dataTrajectory[0,:]).type(torch.FloatTensor)
        dataSection = dataTrajectory[ii:ii+n_steps,:]
        if len(latent_trajectory) > 0:
            latentSection = latent_trajectory[ii:ii+n_steps,:]
        for jj in range(m):
            traj = model.generate_trajectory(x0,n_steps*dt, dt)
            if len(latent_trajectory) == 0:
                mse = ((traj[:n_steps,:] - dataSection) ** 2).sum(axis=-1)
            else:
                proj = mapping(traj)
                mse = ((proj[:n_steps,:] - latentSection)**2).sum(axis=-1)
            weights = np.arange(1,len(mse) + 1)
            fullMSE = np.cumsum(mse)/weights 
            cumMSEs.append(fullMSE)
    
    cumMSEs = np.vstack(cumMSEs)
    avgMSE = np.nanmean(cumMSEs,axis=0)
    sdMSE = np.nanstd(cumMSEs,axis=0)

    return avgMSE,sdMSE

    








