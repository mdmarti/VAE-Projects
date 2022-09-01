import numpy as np
import torch
from sklearn.decomposition import PCA 
import umap
import seaborn as sns
import matplotlib.pyplot as plt
import os

def z_plots(model=None, loader=None):

	'''
	we want to understand what our latent space is doing. therefore we will take in a model and a dataloader, and then look at trajectories!
	inputs:
	- model: a model
	- dataloader: a dataloader
	'''

	print('gettin latents')

	latents = model.get_latent(loader)

	
	stacked_for_transforms = np.vstack(latents)
	#print(stacked_for_transforms.shape)
	

	mean_per_dim = np.mean(stacked_for_transforms,axis=1,keepdims=True)

	covar = (stacked_for_transforms - mean_per_dim).T @ (stacked_for_transforms - mean_per_dim)

	sds = np.sqrt(np.diag(covar))
	denom = sds[:,None] @ sds[None,:]

	corr = covar/denom 

	_, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(6,12))

	sns.heatmap(data=covar,ax=ax1,cbar_kws={'label':'variance/covariance'})
	sns.heatmap(data=corr,ax=ax2,vmin=0.0,vmax=1.0,cbar_kws={'label':'correlation'})

	ax1.set_title('Covariance between latent dims')
	ax2.set_title('Correlation between latent dims')
	ax1.set_ylabel('latent dims')
	ax1.set_xlabel('also latent dims')

	ax2.set_xlabel('also latent dims')


	plt.tick_params(
    	axis='both',          # changes apply to the both-axes
    	which='both',      # both major and minor ticks are affected
    	bottom=False,      # ticks along the bottom edge are off
    	top=False,         # ticks along the top edge are off
    	labelbottom=False) # labels along the bottom edge are off

	plt.savefig(os.path.join(model.save_dir, 'corr_and_cov.png'))
	plt.close('all')

	max_length = max(map(len, latents))

	latents = list(map(lambda x: x if x.shape[0] == max_length else \
								np.vstack((x,np.ones((max_length - x.shape[0],x.shape[1]))*np.nan)),latents))

	stacked_lats_traj = np.stack(latents,axis=0)
	mean_traj = np.nanmean(stacked_lats_traj,axis=1)
	sd_traj = np.nanstd(stacked_lats_traj,axis=1)

	time = np.array(range(1,mean_traj.shape[0]+1))/44100
	for ii in range(mean_traj.shape[1]):
			c = mean_traj[:,ii]
			cs = sd_traj[:,ii]
			ax=plt.gca()
			plt.plot(time,c,'b-',label='daily_mean')
			plt.fill_between(time,c - cs,c+cs,color='b',alpha=0.2)
			plt.ylabel('latent value')
			plt.xlabel('Time relative to motif onset')
			plt.savefig(os.path.join(model.save_dir,'component_' + str(ii+1) + '_voc_trace.png'))

			plt.close('all')

	return

def lookin_at_latents(model=None,loader=None):

	'''
	we want to understand what our latent space is doing. therefore we will take in a model and a dataloader, and then look at trajectories!
	inputs:
	- model: a model
	- dataloader: a dataloader
	'''
	print('gettin latents')

	latents = model.get_latent(loader)

	stacked_for_transforms = np.vstack(latents)

	l_umap = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, random_state=42)
	l_pca = PCA()

	print('Fitting UMAP and PCA')
	latents_umap = l_umap.fit_transform(stacked_for_transforms)
	latents_pca = l_pca.fit_transform(stacked_for_transforms)
	print('DONE')
	sample_inds = np.random.choice(len(latents), 2)

	fig1 = plt.figure(num=1,figsize=[8.5,11])
	ax1 = plt.gca()
	fig2 = plt.figure(num=2,figsize=[8.5,11])
	ax2 = plt.gca()
	fig3 = plt.figure(num=3,figsize=[8.5,11])
	ax3 = plt.gca()
	
	
	print('plotting everything!')
	background_umap = sns.scatterplot(x=latents_umap[:,0],y=latents_umap[:,1],color='k',ax=ax1)
	background12_pca = sns.scatterplot(x=latents_pca[:,0],y=latents_pca[:,1],color='k',ax=ax2)
	background34_pca = sns.scatterplot(x=latents_pca[:,2],y=latents_pca[:,3],color='k',ax=ax3)

	for s in sample_inds:
		tmp_latents = latents[s]

		umap_tmp = l_umap.transform(tmp_latents)
		pca_tmp = l_pca.transform(tmp_latents)

		sns.lineplot(x=umap_tmp[:,0],y=umap_tmp[:,1],ax=ax1)
		sns.lineplot(x=pca_tmp[:,0],y=pca_tmp[:,1],ax=ax2)
		sns.lineplot(x=pca_tmp[:,2],y=pca_tmp[:,3],ax=ax3)

	ax1.set_xlabel('UMAP 1')
	ax1.set_ylabel('UMAP 2')

	ax2.set_xlabel('PCA 1')
	ax2.set_ylabel('PCA 2')

	ax3.set_xlabel('PCA 3')
	ax3.set_ylabel('PCA 4')

	fig1.savefig(os.path.join(model.save_dir,'umap_trajectories.png'))
	fig2.savefig(os.path.join(model.save_dir,'pca_12_trajectories.png'))
	fig3.savefig(os.path.join(model.save_dir,'pca_34_trajectories.png'))

	plt.close('all')
	return 


if __name__ == '__main__':

	pass