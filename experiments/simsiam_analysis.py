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
	stacked_for_transforms
	#print(stacked_for_transforms.shape)
	

	mean_per_dim = np.mean(stacked_for_transforms,axis=1,keepdims=True)

	covar = (stacked_for_transforms - mean_per_dim).T @ (stacked_for_transforms - mean_per_dim)

	sds = np.sqrt(np.diag(covar))
	var_ax = sds > 20
	denom = sds[:,None] @ sds[None,:]

	corr = covar/denom 
	print(sum(var_ax))
	active_var = covar[var_ax,var_ax]
	active_corr = corr[var_ax,var_ax]
	print("Covariance matrix \n")
	print(active_var.shape)
	print("\n correlation matrix \n")
	print(active_corr.shape)
	

	#_, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(15,6))

	sns.heatmap(data=active_var,cbar_kws={'label':'variance/covariance'})
	plt.savefig(os.path.join(model.save_dir, 'cov.png'))
	plt.close('all')
	sns.heatmap(data=active_corr,vmax=1.0,cbar_kws={'label':'correlation'})
	plt.savefig(os.path.join(model.save_dir, 'corr.png'))
	plt.close('all')

	corr_inds = []
	for ii in range(corr.shape[0]):

		corr_inds_dim = corr[ii,:] >= 0.9
	
		corr_inds.append(corr_inds_dim)

	
	max_length = max(map(len, latents))
	latents = list(map(lambda x: x if x.shape[0] == max_length else \
								np.vstack((x,np.ones((max_length - x.shape[0],x.shape[1]))*np.nan)),latents))

	stacked_lats_traj = np.stack(latents,axis=0)
	mean_traj = np.nanmean(stacked_lats_traj,axis=0)
	sd_traj = np.nanstd(stacked_lats_traj,axis=0)

	'''
	time = np.array(range(1,mean_traj.shape[0]+1))/loader.dataset.p['window_overlap']
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
	'''
	tmp_inds = corr_inds[0]
	neg_inds = 1 - tmp_inds
	
	n_dims_big = sum(neg_inds)
	n_dims_small = sum(tmp_inds)
	print(stacked_lats_traj.shape)

	block1 = stacked_lats_traj[:,:,tmp_inds]
	block2 = stacked_lats_traj[:,:,neg_inds]

	#print(block1)
	print(block1.shape)
	print(block2.shape)
	
	coord_1s = np.nanmean(block1,axis=2)
	coord_2s = np.nanmean(block2,axis=2)

	bg1,bg2 = coord_1s[:],coord_2s[:]
	
	mean_c1, mean_c2 = np.nanmean(mean_traj[:,tmp_inds],axis=-1), np.nanmean(mean_traj[:,neg_inds],axis=-1)

	### 
	# something is going on here figure it out later
	#####
	time = np.array(range(1,len(mean_c1) + 1))
	_,(ax1,ax2) = plt.subplots(nrows=2,ncols=1,figsize=(20,20))

	#bg = sns.scatterplot(x=coord_1s.flatten(),y=coord_2s.flatten(),label='all points',alpha=0.5,size=0.5,ax=ax)

	for ii in range(n_dims_big):
		tmp_trajs = block2[:,:,ii]
		mean_tmp = np.nanmean(tmp_trajs,axis=0)
		sns.lineplot(x=time,y=mean_tmp,ax=ax1)

	for ii in range(n_dims_small):
		tmp_trajs=block1[:,:,ii]
		mean_tmp = np.nanmean(tmp_trajs,axis=0)
		sns.lineplot(x=time,y=mean_tmp,ax=ax2)
	#sns.lineplot(x=time,y=mean_c2,color='k',marker='o',ax=ax2)
	#for ii in sample_inds:
	#	sns.lineplot(x=coord_1s[ii,:],y=coord_2s[ii,:],ax=ax)
	#plt.fill_between(time,mean_trajs - sd_trajs,mean_trajs+sd_trajs,color='b',alpha=0.2)
	ax1.set_ylabel('Average trajs in block 1')
	ax2.set_ylabel('averaged trajs in block 2')
	ax2.set_xlabel('time')
	#ax1.set_ylim((-1,1))
	#ax2.set_ylim((-0.01,0.01))
	#plt.xlabel('Time')
	plt.savefig(os.path.join(model.save_dir,'correlated_latent_components_time.png'))
	plt.close('all')

	_, axs = plt.subplots(nrows=8,ncols=8,figsize=(40,40))

	#print(axs)
	axs = axs.reshape(-1)

	index = 0
	samples = np.random.choice(len(latents),9)
	print('sampling latents')
	corr_inds = []
	sizes = []
	print('getting new inds')
	for ii in range(mean_traj.shape[-1]):
		
		l = mean_traj[:,ii]
		print(sum(l))
		time_mean = np.array(range(1,len(l) + 1))

		#for l in latents:
		#	traj_dims = l[:,ii]
		#	time = np.array(range(1,len(l) + 1))

			#traj_dims = traj_dims[:,ii]
		for s in samples:
			samp = latents[s]
			#print(samp.shape)
			time = np.array(range(1,len(samp) + 1))
			sns.lineplot(x=time,y=samp[:,ii],ax = axs[ii])
		sns.lineplot(x=time_mean,y=l,color='k',ax=axs[ii])
		

	plt.savefig(os.path.join(model.save_dir,'all_samples_each_component.png'))
	plt.close('all')

	_, axs = plt.subplots(nrows=3,ncols=3,figsize=(30,30))

	#print(axs)
	print('plotting all samples, each component')
	axs = axs.reshape(-1)
	print(corr_inds)

	#min_ind = np.argmin(sizes)
	for ii,s in enumerate(samples):
		samp = latents[s]
		time = np.array(range(1,len(samp) + 1))
		#min_ind = np.argmin(np.sum(samp[:,corr_inds],axis=0))
		#min_traj = samp[:,corr_inds][:,min_ind]
		for c in range(samp.shape[-1]):
			sns.lineplot(x=time,y=samp[:,c],ax=axs[ii])

	plt.savefig(os.path.join(model.save_dir,'all_components_each_sample_sub.png'))
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

	l_umap = umap.UMAP(n_components=2, n_neighbors=40, min_dist=1e-20, random_state=42)
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