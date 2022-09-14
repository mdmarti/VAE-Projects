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

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir) 
from ava.preprocessing.utils import get_spec


sns.set()
sns.set_context("talk")
def plot_trajectories_umap_and_coords(model,loader,n_samples=5):

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

	spec_cmap = sns.color_palette("mako", as_cmap=True)

	tmpdl = DataLoader(loader.dataset, batch_size=1, \
			shuffle=False, num_workers=loader.num_workers)
	tmpdl.dataset.train_augment=False
	samples = np.random.choice(len(loader.dataset),n_samples)

	specs,files,ons,offs = tmpdl.dataset.__getitem__(samples, seed=None, shoulder=0.05, \
		return_seg_info=True)

	fig, axs = plt.subplots(nrows=2,ncols=n_samples)

	latents = []
	for ind,song in enumerate(specs):
		
		song = torch.stack(song,axis=0).unsqueeze(1).to(model.device)
		print(song.shape)
		onset = ons[ind]
		offset=offs[ind]
		fn = files[ind]
		
		fs, audio = wavfile.read(fn) 
		with torch.no_grad():
				z = model.encoder.encode(song)
				z = z/torch.norm(z,dim=-1,keepdim=True)

		z = z.detach().cpu().numpy()


		latents.append(z)

		target_times = np.linspace(onset, offset, \
						p['num_time_bins'])
		spec, flag = get_spec(max(0.0, onset-shoulder), \
						offset+shoulder, audio, p, \
						fs=fs, target_times=target_times)

		curr_axs = axs[:,ind]

		sns.heatmap(np.flipud(spec),vmin=0.0,cmap=spec_cmap,ax=curr_axs[0])
		time = np.array(range(1,len(z) + 1))

		for dim in range(z.shape[-1]):
			time = np.array(range(1,len(z) + 1))
			#min_ind = np.argmin(np.sum(samp[:,corr_inds],axis=0))
			#min_traj = samp[:,corr_inds][:,min_ind]
			print(np.sum(z[:,dim]))
			if np.sum(z[:,dim]) >= len(z)/4:
				sns.lineplot(x=time,y=z[:,dim] - np.mean(z[:,dim]),ax=curr_axs[1])


		plt.savefig(os.path.join(model.save_dir,'components_specs_plot.png'))

		plt.close('all')

	return