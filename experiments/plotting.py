import matplotlib

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import os
import inspect
import sys 
from scipy.io import wavfile
import umap

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir) 
from ava.preprocessing.utils import get_spec


sns.set()
sns.set_context("talk")

def plot_trajectories_umap_and_coords(model,loader,n_samples=7,fn=''):

	p = {
		'min_freq': 10, # minimum frequency
		'max_freq': 22000, #maximum frequency
		'nperseg': 512, # FFT, length of Hann window
		'noverlap': 256, # FFT, overlap in sequences
		'spec_min_val': 2.0, # minimum log-spectrogram value
		'spec_max_val': 7.0, # maximum log-spectrogram value
		'fs': 44100, # audio samplerate
		'get_spec': get_spec, # figure out what this is
		'min_dur': 0.35, #0.015, # minimum syllable duration
		'max_dur': 1.1, #0.25, #maximum syllable duration
		'smoothing_timescale': 0.007, #amplitude
		'temperature': 0.5, # softmax temperature parameter
		'softmax': False, # apply softmax to frequency bins to calculate amplitude
		'th_1': 2.25, # segmenting threshold 1
		'th_2': -1, # segmenting threshold 2
		'th_3': 4.5, # segmenting threshold 3
		'window_length': 0.12, # spec window, in s
		'window_overlap':0.11, # overlap between spec windows, in s
		'num_freq_bins': 128,
		'num_time_bins': 128,
		'mel': True, # Frequency spacing: mel-spacing for birbs
		'time_stretch': True, #are we warping time?
		'within_syll_normalize': False, #normalize within syllables?
		'real_preprocess_params': ('min_freq', 'max_freq', 'spec_min_val', \
			'spec_max_val'), #I'm sure this does somethinglatent_path
		'int_preprocess_params': tuple([]), #i'm ALSO sure this does something
		'binary_preprocess_params': ('mel', 'within_syll_normalize'), #shrug
        'train_augment': True, # whether or not we are training simsiam
		'max_tau':0.01 # max horizontal time shift for our window
		}
	shoulder = 0.05

	all_latents = model.get_latent(loader)
	stacked_latents = np.vstack(all_latents)
	simsiam_umap = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, random_state=42)
	
	stacked_labels = np.zeros((stacked_latents.shape[0],))

	spec_cmap = sns.color_palette("mako", as_cmap=True)

	tmpdl = DataLoader(loader.dataset, batch_size=1, \
			shuffle=False, num_workers=loader.num_workers)
	tmpdl.dataset.train_augment=False
	samples = np.random.choice(len(loader.dataset),n_samples)

	specs,files,ons,offs = tmpdl.dataset.__getitem__(samples, seed=None, shoulder=0.05, \
		return_seg_info=True)

	fig, axs = plt.subplots(nrows=3,ncols=n_samples,figsize=(75,50))
	#fig = plt.figure(figsize=(75,50))
	latents = []
	latent_labels = []

	for ind,song in enumerate(specs):
		
		song = torch.stack(song,axis=0).unsqueeze(1).to(model.device)
		print(song.shape)
		onset = ons[ind]
		offset=offs[ind]
		fn = files[ind]
		
		fs, audio = wavfile.read(fn) 
		with torch.no_grad():
				z = model.encoder.encode(song)
				#z = z/torch.norm(z,dim=-1,keepdim=True)

		z = z.detach().cpu().numpy()

		target_times = np.linspace(onset, offset, \
						p['num_time_bins'])
		spec, flag = get_spec(max(0.0, onset-shoulder), \
						offset+shoulder, audio, p, \
						fs=fs, target_times=target_times)

		if np.max(spec) < 0.1:
				print("remaking spec: too quiet")
				flag = False
		if np.sum(spec) < 2500:
				print('Remaking spec: too quiet')
				flag = False
		while not(flag):
			new_s = np.random.choice(len(loader.dataset),1)[0]
			spec_n,file_n,on_n,off_n = tmpdl.dataset.__getitem__(new_s, seed=None, shoulder=0.05, \
					return_seg_info=True)

			onset = on_n
			offset=off_n
			fn = file_n
		
			fs, audio = wavfile.read(fn)
			spec_n = torch.stack(spec_n,axis=0).unsqueeze(1).to(model.device)
			print(spec_n.shape)
			with torch.no_grad():
				z = model.encoder.encode(spec_n)
				#z = z/torch.norm(z,dim=-1,keepdim=True)

			z = z.detach().cpu().numpy()


			

			target_times = np.linspace(onset, offset, \
						p['num_time_bins'])
			spec, flag = get_spec(max(0.0, onset-shoulder), \
						offset+shoulder, audio, p, \
						fs=fs, target_times=target_times)

			if np.max(spec) < 0.1:
				print("remaking spec: too quiet")
				flag = False
			if np.sum(spec) < 2500:
				print('Remaking spec: too quiet')
				flag = False
			


		print('made spec!')
		print("sum of spec: ", np.sum(spec))
		latents.append(z)
		#ax = fig.add_subplot(3,n_samples,ind+1)
		curr_axs = axs[:,ind]
		latent_labels.append(np.ones((z.shape[0],)) + ind)
		sns.heatmap(np.flipud(spec),vmin=0.0,cmap=spec_cmap,ax=curr_axs[0],cbar=False)
		time = np.array(range(1,len(z) + 1))

		for dim in range(z.shape[-1]):
			time = np.array(range(1,len(z) + 1))
			#min_ind = np.argmin(np.sum(samp[:,corr_inds],axis=0))
			#min_traj = samp[:,corr_inds][:,min_ind]
			#print(np.sum(z[:,dim]))
			#ax = fig.add_subplot(3,n_samples,n_samples + ind+1)
			
			if np.sum(z[:,dim]) >= 8:
				sns.lineplot(x=time,y=z[:,dim] - np.mean(z[:,dim]),ax=curr_axs[1])
		
	latents = np.vstack(latents)
	latent_labels = np.hstack(latent_labels)
	stacked_latents = np.vstack([stacked_latents,latents])
	stacked_labels = np.hstack([stacked_labels,latent_labels])
	print('Fitting UMAP')

	umapped_latents = simsiam_umap.fit_transform(stacked_latents)
	#projected_trajectory = simsiam_umap.transform(z)
	for ind,song in enumerate(specs):
		curr_axs = axs[:,ind]
		time = np.array(range(1,len(umapped_latents[stacked_labels==(ind+1),0]) + 1))
		
		#ax = fig.add_subplot(3,n_samples,2*n_samples + 1 + ind)
		bg = curr_axs[2].scatter(umapped_latents[stacked_labels==0,0],umapped_latents[stacked_labels==0,1],color='k',alpha=0.01,s=0.5)
		#trajectory = sns.lineplot(x=umapped_latents[stacked_labels==(ind+1),0],y=umapped_latents[stacked_labels==(ind + 1),1],ax=curr_axs[2],sort=False,color='r')
		trajectory = curr_axs[2].scatter(umapped_latents[stacked_labels==(ind+1),0],umapped_latents[stacked_labels==(ind + 1),1],\
										c=time,cmap='flare')

	plt.show()
<<<<<<< HEAD
	plt.savefig(os.path.join(model.save_dir,'components_specs_plot_scatter.png'))
=======
	plt.savefig(os.path.join(model.save_dir,fn + 'components_specs_plot_scatter_normed.png'))
>>>>>>> 7b21e0b09cc432f3647b8981261eca151e905ed5

	plt.close('all')

	print('fitting 3d')
	simsiam_umap3d = umap.UMAP(n_components=3, n_neighbors=20, min_dist=0.1, random_state=42)
	umapped_latents = simsiam_umap3d.fit_transform(stacked_latents)

	fig = plt.figure(figsize=(75,20))

	for ind,song in enumerate(specs):
		#curr_axs = axs[:,ind]
		time = np.array(range(1,len(umapped_latents[stacked_labels==(ind+1),0]) + 1))
		
		ax = fig.add_subplot(1,n_samples,1 + ind,projection='3d')
		bg = ax.scatter(umapped_latents[stacked_labels==0,0],umapped_latents[stacked_labels==0,1],umapped_latents[stacked_labels==0,2],color='k',alpha=0.01,s=0.5)
		#trajectory = sns.lineplot(x=umapped_latents[stacked_labels==(ind+1),0],y=umapped_latents[stacked_labels==(ind + 1),1],ax=curr_axs[2],color='r')
		trajectory = ax.scatter(umapped_latents[stacked_labels==(ind+1),0],umapped_latents[stacked_labels==(ind + 1),1],\
										umapped_latents[stacked_labels==(ind + 1),2],c=time,cmap='flare')

		plt.show()

	plt.savefig(os.path.join(model.save_dir,fn + 'components_specs_plot_3d.png'))

	plt.close('all')

	return