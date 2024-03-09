import numpy as np
from ripser import ripser
from sklearn.linear_model import LinearRegression

def cohomology_analysis(model,data):

    """
    do some cohomology analysis
    """

def between_latent_comparison(latents1,latents2):

    """
    compare two sets of latents (can be latents and real data, too)
    """

    lr = LinearRegression()
    lr.fit(latents1,latents2)
    latents2Hat = lr.predict(latents1)

    sstot = ((latents1 - np.nanmean(latents1,axis=0))**2).sum(axis=0)
    ssresid = ((latents1 - np.nanmean(latents1,axis=0))**2).sum(axis=0)
    r2 = 1 - sstot/ssresid 
    r2 = r2.sum()/len(r2) # average r^2 across dimensions

    return r2


