import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir) 

import fire
from VAE_Projects.models.models import encoder,decoder,VAE_Base,SmoothnessPriorVae,ReconstructTimeVae
from VAE_Projects.models.utils import rbf_dot,mmd_fxn,numpy_to_tensor
import matplotlib.pyplot as plt
from colour import Color
import numpy as np
import os
import cv2
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA, FastICA
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_moons
from sklearn.linear_model import LinearRegression
import scipy

import torch
import umap

from torch.utils.data import Dataset, DataLoader
from torch.distributions import LowRankMultivariateNormal
from joblib import Parallel, delayed
import numpy as np
import os
import numba


from ava.data.data_container import DataContainer
from ava.models.vae import X_SHAPE, VAE
from ava.models.vae_dataset import get_syllable_partition, \
	get_syllable_data_loaders
from ava.preprocessing.preprocess import process_sylls, \
	tune_syll_preprocessing_params
from ava.preprocessing.utils import get_spec
from ava.segmenting.refine_segments import refine_segments_pre_vae
from ava.segmenting.segment import tune_segmenting_params
from ava.segmenting.amplitude_segmentation import get_onsets_offsets
from ava.segmenting.template_segmentation import get_template, segment_files, \
	clean_collected_segments, segment_sylls_from_songs, read_segment_decisions
from ava.models.window_vae_dataset import get_window_partition, \
				get_fixed_window_data_loaders, get_fixed_ordered_data_loaders_motif

FS = 42000

##### TO-DO ##################
# for all below points: just use a subset of data!!!! We don't need it all!!!
# just get representative plots fool!
# make trajectory plots overlaid onto group UMAP
# make clean predicted vs. real UMAP
# MAKE CLEAN QUIVER PLOT
# drift & noise magnitude per day
# simulated & reconstructed specs
# quiver drift where drift vec is over days
# FILTER: what we want is to delete silence then allow_pickle
# if time after: increase window size, overlap
# plot integrated trajectory on top of this

def pca_analysis(model,loader):

	latents = np.vstack(model.get_latent(loader))

	#print(latents.shape)
	latent_pca = PCA()

	transformed_latents = latent_pca.fit_transform(latents)

	v_explained_cumu = np.cumsum(latent_pca.explained_variance_ratio_)

	ind = np.where(v_explained_cumu >= 0.99)[0][0]

	print('Dimensionality of latents: {:d}'.format(ind + 1))

	return transformed_latents, latents

def smoothness_analysis(model,loader):

	## to implement: need to include loaders for all days.
	## once you do so, implement MMD analysis for all days. visualize using UMAP, colored by MMD. 
	## look at MMD distribution for each part of trajectory. maybe weight reconstruction by MMD also? 
	## look at which vocalizations change the least (by MMD), which change the most positively, 
	## which change the most negatively

	latents = model.get_latent(loader)

	max_len_traj = max(list(map(len,latents)))

	mean_traj_day = np.empty((len(latents),max_len_traj,32))

	mean_traj_day[:,:,:] = np.nan

	for ind,traj in enumerate(latents):

		mean_traj_day[ind,0:traj.shape[0],:] = traj

	mean_traj = np.nanmean(mean_traj_day,axis=0)

	mean_recon_day = model.decoder.decode(numpy_to_tensor(mean_traj).to(model.device))

	frames = []
	for spec in mean_recon_day:
		frames.append(spec)

	stitcher = cv2.createStitcher()
	(status,stitched) = stitcher.stitch(frames)


	ax = plt.gca()
	im = ax.imshow(stitched,origin='lower',vmin=0,vmax=1)

	plt.savefig('mean_image_.png')

	return mean_traj, stitched

def model_comparison_umap(vanilla,smoothprior,time_recon,loader,n_samples = 5):

	print('getting vanilla latents')
	latents_vanilla = vanilla.get_latent(loader)
	print('getting smoothprior latents')
	latents_smoothprior = smoothprior.get_latent(loader)
	print('getting time latents')
	latents_time = time_recon.get_latent(loader)

	stacked_vanilla = np.vstack(latents_vanilla)
	stacked_smooth = np.vstack(latents_smoothprior)
	stacked_time = np.vstack(latents_time)
	print(stacked_vanilla.shape)
	print(stacked_smooth.shape)
	print(stacked_time.shape)

	latents_stacked = np.vstack([stacked_vanilla,stacked_smooth,stacked_time])


	joint_umap = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, random_state=42)

	print('fittin umap')
	stacked_transformed = joint_umap.fit_transform(latents_stacked)
	print('done!')
	ax = plt.gca()

	vanilla_shape = len(latents_vanilla)
	smooth_shape = len(latents_smoothprior)
	time_shape = len(latents_time)

	vanilla_samps = np.random.choice(vanilla_shape,n_samples,replace=False)
	smooth_samps = np.random.choice(smooth_shape,n_samples,replace=False)
	time_samps = np.random.choice(time_shape,n_samples,replace=False)

	print('plotting vanilla latents')
	bg = ax.scatter(stacked_transformed[:,0],stacked_transformed[:,1],s=0.25,alpha=0.05,color='k')
	for s in vanilla_samps:
		tmp_samp = latents_vanilla[s]
		tmp_umap = joint_umap.transform(tmp_samp)
		ax.plot(tmp_umap[:,0],tmp_umap[:,1], color='r')

	plt.savefig(os.path.join(vanilla.save_dir,'vanilla_latent_samples.png'))
	plt.close('all')
	ax = plt.gca()
	print('plotting smooth prior latents')
	bg = ax.scatter(stacked_transformed[:,0],stacked_transformed[:,1],s=0.25,alpha=0.05,color='k')
	for s in smooth_samps:
		tmp_samp = latents_smoothprior[s]
		tmp_umap = joint_umap.transform(tmp_samp)
		ax.plot(tmp_umap[:,0],tmp_umap[:,1], color='r')

	plt.savefig(os.path.join(smoothprior.save_dir,'smoothprior_latent_samples.png'))
	plt.close('all')
	ax = plt.gca()
	print('plotting time recon latents')
	bg = ax.scatter(stacked_transformed[:,0],stacked_transformed[:,1],s=0.25,alpha=0.05,color='k')
	for s in time_samps:
		tmp_samp = latents_time[s]
		tmp_umap = joint_umap.transform(tmp_samp)
		ax.plot(tmp_umap[:,0],tmp_umap[:,1], color='r')

	plt.savefig(os.path.join(time_recon.save_dir,'timerecon_latent_samples.png'))
	plt.close('all')

	return





def bird_model_script(vanilla_dir='',smoothness_dir = '',time_recondir = '',datadir='',):

########## Setting up directory lists: separate into train, test dirs
############# Expected File Structure

# Root (code, stuff, etc)
# |
# | --- Animal 1
# |	  |
# |	  | - day 1
# |	  |   | - data dirs
# |	  |
# |	  | - day 2
# |		  | - data dirs
# |
# | --- Animal 2
# | ...

	# here is where the data are stored
	datadir = '/home/mrmiews/' #'/media/mrmiews/Storage for Data/'

	# here are the days we are using to train our model. Each day has .wav files
	# containing song
	trainDays = ['36','37','38','39','40','41','42','43','44','45','46','47','48','49',\
				'50','51','52','53','54','55','56','57','58','62','63','64','65','66','67','68','69',\
				'70','71','72','73','74','75','76','77','78','79','80','81','82','83',\
				'84','85','86','87','88','89','90','91','92','93','94','95','96',\
				'97','98','99','100','102','103','104','105','106','107','108','109','110','111','112','115',\
				'118','120','122','124','126','128','130','131','132','133','134',\
				'135','136','137','138','139','140'] #['UNDIRECTED_11092020','DIRECTED_11092020']#,,

	# subset for traininng
	realTrainDays = ['70','71','72','73','74','75','76','77','78','79','80','81','82','83']
	#['40','45','55','74','79','87','97','105','112','120','130','135','138','140']
	# another subset. why this is necessary? who knows
	songDays = ['44','45','46','47','48','49','50','51','52','53','54','55','56','57','58']
	# another subset. Could have just used one, but apparently I didn't feel like it!
	plotDays = ['44','45','46','52','53','54','55','56','57','58','62','63','64','65',\
	'77','78','79','80','81','88','89','90','91','92','93','94','95','96',\
	'97','98','109','110','111','112','115','124','130','134','139']

	# bird directory
	dsb = ['org384']
	# latent dimension
	z_dim = 32

	# these were actually important - used for R-VAE experiments so we could have CLEAN data
	adult_audio_dirs = ['/home/mrmiews/Desktop/Pearson_Lab/bird_data/blk417_tutor/motif_audio_tutor',
						'/home/mrmiews/Desktop/Pearson_Lab/bird_data/blk411_tutor/motif_audio_tutor']
	adult_motif_dirs = ['/home/mrmiews/Desktop/Pearson_Lab/bird_data/blk417_tutor/motif_segs',
						'/home/mrmiews/Desktop/Pearson_Lab/bird_data/blk411_tutor/motif_segs']


	# directory names for each data source. audio, segment (ROIs within audio), specs (unused), projections (unused)
	dsb_audio_dirs = [os.path.join(datadir,animal,'audio',day) for animal in dsb for day in realTrainDays]

	dsb_segment_dirs = [os.path.join(datadir,animal,'syll_segs',day) for animal in dsb for day in realTrainDays]

	dsb_spec_dirs = [os.path.join(datadir,animal,'syll_specs',day) for animal in dsb for day in realTrainDays]

	dsb_proj_dirs = [os.path.join(datadir,animal,'syll_projs',day) for animal in dsb for day in realTrainDays]

	# params for model. Why is this here? it should be lower
	plots_dir = os.path.join(root,'plots')

	template_b1_dir = os.path.join(datadir, 'org384','org384_tutor_motifs')


	####################################
	# 0.5) Define segmenting parameters #
	#####################################

	segment_params = {
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
		'window_length': 0.10, # spec window, in s
		'window_overlap':0.09, # overlap between spec windows, in s
		'algorithm': get_onsets_offsets, #finding syllables
		'num_freq_bins': X_SHAPE[0],
		'num_time_bins': X_SHAPE[1],
		'mel': True, # Frequency spacing: mel-spacing for birbs
		'time_stretch': True, #are we warping time?
		'within_syll_normalize': False, #normalize within syllables?
		'real_preprocess_params': ('min_freq', 'max_freq', 'spec_min_val', \
			'spec_max_val'), #I'm sure this does somethinglatent_path
		'int_preprocess_params': tuple([]), #i'm ALSO sure this does something
		'binary_preprocess_params': ('mel', 'within_syll_normalize'), #shrug
		}

	#template_b1 = get_template(template_b1_dir,segment_params)
	#result_b1 = segment_files(dsb_audio_dirs_tutor,dsb_song_dirs,template_b1,\
	#		segment_params,num_mad=5.0,n_jobs=8)
	#clean_collected_segments(result_b1,dsb_song_dirs,dsb_song_dirs,segment_params)
	#segment_params = tune_segmenting_params(dsb_audio_dirs_tutor, segment_params)

#############################
# 1) Amplitude segmentation #
#############################
	#segment_params = tune_segmenting_params(dsb_audio_dirs,segment_params)

	#from ava.segmenting.segment import segment
	#for audio_dir, segment_dir in zip(dsb_audio_dirs, dsb_segment_dirs):
		#segment(audio_dir, segment_dir, segment_params)


	# this is using full motifs, so it should be fine. if you want to use the actual youth song,
	# change the get_window_partition bit to include freshly segmented data
	#loaders = get_fixed_window_data_loaders(test_part, segment_params)
	motif_part = get_window_partition(dsb_audio_dirs,dsb_segment_dirs,0.8)
	#motif_part['test'] = motif_part['train']
	print('getting prediction loader')
	loaders_for_prediction = get_fixed_ordered_data_loaders_motif(motif_part,segment_params)
	# this is used for the shotgun VAE, as opposed to the shotgun-dynamics VAE
	#partition = get_window_partition(dsb_audio_dirs, dsb_segment_dirs, split)
	#loaders = get_fixed_window_data_loaders(partition, segment_params)

#############################
# 1) Train model            #
#############################
	if vanilla_dir != '':
		if not os.path.isdir(vanilla_dir):
			os.mkdir(vanilla_dir)
		save_file = os.path.join(vanilla_dir,'checkpoint_encoder_300.tar')
		#print(save_file)
		vanilla_encoder = encoder()
		vanilla_decoder = decoder()
		vanilla_vae = VAE_Base(vanilla_encoder,vanilla_decoder,vanilla_dir)

		if not os.path.isfile(save_file):
			print('training vanilla')
			vanilla_vae.train_test_loop(loaders_for_prediction,epochs=151,test_freq=5,save_freq=50,vis_freq=25)
		else:
			print('loading vanilla')
			vanilla_vae.load_state(save_file)
			#vanilla_vae.train_test_loop(loaders_for_prediction,epochs=151,test_freq=5,save_freq=50,vis_freq=25)
			#vanilla_vae.test_epoch(loaders_for_prediction['test'])
			loaders = []
			for ind,day in enumerate(realTrainDays):
				part = get_window_partition([dsb_audio_dirs[ind]],[dsb_segment_dirs[ind]],1.0)
				part['test'] = part['train']
				loader = get_fixed_ordered_data_loaders_motif(part,segment_params)
				loaders.append(loader)
			'''
			for ind, l in enumerate(loaders):
				print('Developmental day {} \n'.format(realTrainDays[ind]))
				_,_ = pca_analysis(vanilla_vae,l['train'])
			'''

			'''
			
			Add in new analyses here!!!!
			'''
	if smoothness_dir != '':
		if not os.path.isdir(smoothness_dir):
			os.mkdir(smoothness_dir)
		save_file = os.path.join(smoothness_dir,'checkpoint_encoder_300.tar')
		smooth_encoder = encoder()
		smooth_decoder = decoder()
		smooth_prior_vae = SmoothnessPriorVae(smooth_encoder,smooth_decoder,smoothness_dir)

		if not os.path.isfile(save_file):
			print('training smooth')
			smooth_prior_vae.train_test_loop(loaders_for_prediction,epochs=151,test_freq=5,save_freq=50,vis_freq=25)
		else:
			print('loading smooth')
			smooth_prior_vae.load_state(save_file)
			#smooth_prior_vae.train_test_loop(loaders_for_prediction,epochs=151,test_freq=5,save_freq=50,vis_freq=25)
			#smooth_prior_vae.test_epoch(loaders_for_prediction['test'])
			'''
			for ind, l in enumerate(loaders):
				print('Developmental day {} \n'.format(realTrainDays[ind]))
				_,_ = pca_analysis(smooth_prior_vae,l['train'])
			'''
	if time_recondir != '':
		if not os.path.isdir(time_recondir):
			os.mkdir(time_recondir)
		save_file = os.path.join(time_recondir,'checkpoint_encoder_300.tar')
		time_encoder = encoder()
		time_decoder = decoder()
		time_vae = ReconstructTimeVae(time_encoder,time_decoder,time_recondir)

		if not os.path.isfile(save_file):
			print('training time')
			time_vae.train_test_loop(loaders_for_prediction,epochs=151,test_freq=5,save_freq=50,vis_freq=25)
		else:
			print('loading time')
			time_vae.load_state(save_file)
			#time_vae.test_epoch(loaders_for_prediction['test'])
			#time_vae.train_test_loop(loaders_for_prediction,epochs=151,test_freq=5,save_freq=50,vis_freq=25)

			'''
			for ind, l in enumerate(loaders):
				print('Developmental day {} \n'.format(realTrainDays[ind]))
				_,_ = pca_analysis(time_vae,l['train'])
			'''

	print('doing model comparison')
	model_comparison_umap(vanilla_vae,smooth_prior_vae,time_vae,loaders_for_prediction['test'])
	'''
	print('umappin')
	umap_latents = np.vstack(all_latents)
	color_inds = np.hstack(color_inds)
	umap_t = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, random_state=42)
	umap_ls = umap_t.fit_transform(umap_latents)

	ax = plt.gca()
	cmin = Color('#2e1e3b')
	cmax = Color('#8bdab3')
	print('plotting')
	#val_range = np.linspace(-1,1,500)
	crange = list(cmin.range_to(cmax,len(np.unique(color_inds))))
	crange = [color.hex_l for color in crange] # list(c1.range_to(c2,len(trainDays)))]
	labels = []
	for ci,c in enumerate(np.unique(color_inds)):
		inds = color_inds == c
		ax.scatter(umap_ls[inds, 0],umap_ls[inds,1],color=crange[ci],alpha=0.01)
		sc = ax.scatter(umap_ls[inds[0],0],umap_ls[inds[0],1],color=crange[ci],alpha=1)
		labels.append(sc)

	ax.legend(labels,names)
	ax.set_xlabel('UMAP dim 1')
	ax.set_ylabel('UMAP dim 2')

	plt.savefig(os.path.join(root,'movie_vaeNormal','umap_of_lats.pdf'))
	plt.close('all')

	print('plotting pairwise')
	fig, axs = plt.subplots(nrows=4,ncols=4)
	fig.set_size_inches(18.5, 10.5, forward=True)

	#plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.8,hspace=0.8)
	l1 = [0,1,11,9]
	l2 = [2,6,7,14]
	labels = []
	for ci,c in enumerate(np.unique(color_inds)):
		inds = color_inds == c
		for a1,ind1 in enumerate(l1):
			for a2,ind2 in enumerate(l2):

				axs[a1,a2].scatter(umap_latents[inds,ind1],umap_latents[inds,ind2],color=crange[ci],alpha=0.05,s=0.1)
				axs[a1,a2].set_xlabel('UMAP dim ' + str(ind1 + 1))
				axs[a1,a2].set_ylabel('UMAP dim ' + str(ind2 + 1))


		sc = axs[a1,a2].scatter(umap_latents[inds[0],ind1],umap_latents[inds[0],ind2],color=crange[ci],alpha=1,s=1)
		labels.append(sc)

	axs[a1,a2].legend(labels,names)
	#fig.tight_layout(pad=5.0)
	plt.savefig(os.path.join(root,'movie_vaeNormal','pairwise_latents.pdf'))
	plt.close('all')

	'''

	print(ev_fname)
	if not os.path.isfile(ev_fname):
		embedding_VAE.train_loop(loaders_for_prediction,epochs=501,test_freq=101,vis_freq=20,save_freq=50)
	else:
		embedding_VAE.load_state(ev_fname)

		embedding_VAE.train_loop(loaders_for_prediction,epochs=500,test_freq=101,vis_freq=20,save_freq=50)


	train_l = loaders_for_prediction['train']

	indices = np.random.choice(np.arange(len(train_l.dataset)),
		size=50,replace=False)
	(ims, days) = train_l.dataset[indices]
	#specs = torch.stack(ims).to(self.device)
	all_latents = []
	with torch.no_grad():
		for ti, traj in enumerate(ims):

			latent_traj = torch.stack(traj).to(torch.device('cuda'))
			full_ims = latent_traj.detach().cpu().numpy()

			f1 ,axs1 = plt.subplots(nrows = np.ceil(full_ims.shape[0]/9).astype(int), ncols = 9)
			#ax = plt.gca()
			#print(axs)
			latents = np.zeros((len(traj),32))
			recons = []
			row_ind = 0
			col_ind = 0
			for iter in range(latent_traj.shape[0]):
				#print(latent_traj.shape)
				if col_ind > 8:
					col_ind = 0
					row_ind += 1
				axs1[row_ind,col_ind].imshow(np.flip(full_ims[iter,:,:],axis=0))
				#axs[row_ind,col_ind].axis('square')
				axs1[row_ind,col_ind].set_axis_off()


				mean_lat, _, _ = embedding_VAE.encode(latent_traj[iter,:,:].unsqueeze(0))

				latents[iter,:] = mean_lat.detach().cpu().numpy()

				recon = embedding_VAE.decode(mean_lat)

				recons.append(np.reshape(recon.detach().cpu().numpy(),(128,128)))
				col_ind += 1
			for ii in range(col_ind,9):

				axs1[row_ind,ii].set_axis_off()

			#f.tight_layout()
			plt.savefig(os.path.join(root,'movie_vaeAR_joint_beta50','trajs','trajectory_sample_' + str(indices[ti]) + '.png'))

			plt.close('all')

			f2 ,axs2 = plt.subplots(nrows = np.ceil(full_ims.shape[0]/9).astype(int), ncols = 9)
			row_ind = 0
			col_ind = 0
			for im in recons:

				if col_ind > 8:
					col_ind = 0
					row_ind += 1
				axs2[row_ind,col_ind].imshow(np.flip(im,axis=0))
				#axs[row_ind,col_ind].axis('square')
				axs2[row_ind,col_ind].set_axis_off()
				col_ind += 1

			for ii in range(col_ind,9):

				axs2[row_ind,ii].set_axis_off()

			plt.savefig(os.path.join(root,'movie_vaeAR_joint_beta50','trajs','trajectory_recon_' + str(indices[ti]) + '.png'))

			plt.close('all')

			all_latents.append(latents)
	'''
	#pca1 = PCA(n_components = 32)
	#assert False
	#pca_lats = np.vstack(all_latents)
	#pca1.fit(pca_lats)
	#pca_lats = pca1.transform(pca_lats)

	#umap_transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, random_state=42)
	#umap_lats = umap_transform.fit_transform(pca_lats)


	vars = np.var(pca_lats,axis=0)

	for plot_ind in range(32):



		for ti in range(len(all_latents)):
			f = plt.figure(plot_ind)
			ax = plt.gca()
			curr_lat = all_latents[ti][:,plot_ind]
			#pca_lat = pca1.transform(curr_lat)[:,plot_ind]
			#curr_lat = np.diff(curr_lat)
			#curr_lat = np.cumsum(np.abs(curr_lat))
			ax.plot(list(range(len(curr_lat))),curr_lat,color='g',alpha=0.8)

			ax.set_xlabel('Time (Samples)')
			ax.set_ylabel('Latent Component ' + str(plot_ind + 1))

			ax.set_title('Traj ' + str(indices[ti]) + ' Latent Component ' + str(plot_ind + 1) + ' Variance: ' + str(vars[plot_ind]))

			plt.savefig(os.path.join(root,'movie_vaeNormal', 'traj_' + str(indices[ti]) + '_latent_component_' + str(plot_ind + 1) + '.png'))

			plt.close('all')
	'''

	'''
	print('Getting latents (shotgun)')
	test_l = embedding_VAE.get_latent(loaders['train'])
	print('getting new pnglatents')
	adult_latents_motif = embedding_VAE.get_latent(loaders_for_prediction['train'])
	'''

	'''
	sel = np.random.permutation(len(trainDays))[:17]
	print('embedding everything')
	umap_latents = []
	all_spec = []
	lp = []
	ai = []
	for ind, audd in enumerate(dsb_audio_dirs_all):
		print('Round ' + str(ind))
		segd = dsb_segment_dirs_all[ind]

		partition = get_window_partition([audd], [segd], 0.5)
		loaders = get_fixed_ordered_data_loaders_motif(partition, segment_params)
		i = 0
		l =  []
		umap_l = []
		specs = []
		for data in loaders['train']:
			tl = []
			dd = []
			for d in data:
				d = d.to(embedding_VAE.device)
				with torch.no_grad():
					mu, _, _ = embedding_VAE.encode(d)
				mu = mu.detach().cpu().numpy()
				tl.append(mu)
				dd.append(d.cpu().detach().numpy())
			if len(tl) > 0:
				tl = np.vstack(tl)
				umap_tl = np.hstack([tl,(np.arange(1,len(tl) + 1,1)/len(tl))[:,np.newaxis]])
				#latents.append(tl)
				dd1 = np.vstack(dd)

			#mu = mu.detach().cpu().numpy()
			#mu = np.hstack([mu,(np.arange(1,len(mu) + 1,1)/len(mu))[:,np.newaxis]])
				#print(i)
				#print(len(tl))
				#print()
				specs.append(dd1)
				l.append(umap_tl)
				i += len(tl)



		latent_inds.append(ind*np.ones((l.shape[0],)))
		if ind > 61 or ind < 11:
			lp.append(l)
			umap_latents.append(np.vstack(l))
			all_spec.append(specs)
			ai.append(ind*np.ones((l.shape[0],)))
		ind += 1

	umap_latents = np.vstack(umap_latents)
	#latents = np.vstack([latents,test_l])
	latent_inds = np.hstack(ai)
	#latent_inds = np.hstack([latent_inds,-1*np.ones(test_l.shape[0],)])
	#all_spec = np.vstack(all_spec)
	all_inds = np.hstack(ai)
	lp = np.vstack(lp)
	print(latents.shape)
	print(all_spec.shape)
	print('running UMAP')
	umap_transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, \
	metric=time_weighted_distance, random_state=42)
	ep = umap_transform.fit_transform(umap_latents)
	#ep = umap_transform.transform(lp)
	new_spec,new_lat,new_ind,new_e,bounds = make_cleaned_plot(umap_latents,ep,all_spec,ai,lp,[]])
	'''
	latent_path = os.path.join(root,'latent_nn_test','latent_files')
	fnames_all = []
	fnames_plot = []
	print('pickling')
	for ind, (audio,segment) in enumerate(zip(dsb_audio_dirs_all,dsb_segment_dirs_all)): #enumerate(zip(adult_audio_dirs,adult_motif_dirs))
		#print(trainDays[ind])

		#d_int = int(trainDays[ind])
		if True: #int(trainDays[ind]) <= 150:
			just_read = True
			latent_fnames_stuff = latents_to_pickle([], [],\
							os.path.join(latent_path,trainDays[ind]),n_per_file=30,just_read = just_read)
			#latent_fnames_stuff = latents_to_pickle([], [],\
			#				os.path.join(latent_path,str(190)),n_per_file=30,just_read = just_read)
		else:
			just_read = False
			tmp_part = get_window_partition([audio],[segment],1.0)
			tmp_part['test'] = None
			tmp_loaders = get_fixed_ordered_data_loaders_motif(tmp_part,segment_params)
			d_int = 190 # remove once this works for adults, fix embedding

		# fake adult day: day 190!
			####################################################################
			##### Fix this part: get latents so that you can  #################
			#### pick certain ones aka filter silence and noise ################
			###################################################################


			latents_tmp = embedding_VAE.get_latent(tmp_loaders['train'])
		#	et = [umap_transform.transform(np.hstack([l,(np.arange(1,len(l) + 1,1)/len(l))])) for l in latents_tmp]
		#	em_tmp = umap_transform.transform(lt)
		#	new_spec,new_lat,new_ind,new_e,bounds = make_cleaned_plot([],[],[],\
		#			np.ones(len(latents_tmp)),latents_tmp,et,bounds)
			latent_fnames_stuff = latents_to_pickle(latents_tmp, list(d_int * np.ones((len(latents_tmp),))),\
							os.path.join(latent_path,str(190)),n_per_file=30,just_read = just_read)
			#latent_fnames_stuff = latents_to_pickle(latents_tmp, list(d_int * np.ones((len(latents_tmp),))),\
			#				os.path.join(latent_path,trainDays[ind]),n_per_file=30,just_read = just_read)
		fnames_all += latent_fnames_stuff['latent_fnames']
		if trainDays[ind] in plotDays:
			fnames_plot += latent_fnames_stuff['latent_fnames']

	#print(fnames_all)
	#adult_latents_motif = [torch.from_numpy(x).double() for x in adult_latents_motif]
	#print(adult_latents_motif.shape)

	#day_vec_adults = [1 for i in range(len(adult_latents_motif))]
	#motif_vec_adults = [1 for i in range(len(adult_latents_motif))]

	##### Just train on whole thing first, worry about train-test split later #####


	print('Making Latent Loaders')
	latents_ds = latents_motif_ds(fnames_all,\
									transform=numpy_to_tensor)
	latents_dl = DataLoader(latents_ds, batch_size=1, \
			shuffle=True, num_workers=os.cpu_count()-2)
	latents_loaders = {'trashapein': latents_dl, 'test':latents_dl}
	latents_ds_pl = latents_motif_ds(fnames_plot,\
									transform=numpy_to_tensor)
	latents_dl_pl = DataLoader(latents_ds_pl, batch_size=1, \
			shuffle=True, num_workers=os.cpu_count()-2)
	latents_pl_loaders = {'train':latents_dl_pl,'test':latents_dl_pl}

	latent_model = nn_predict(save_dir = os.path.join(root,'latent_nn_real'),z_dim=32)

	model_state = os.path.join(root,'latent_nn_real','checkpoint_030.tar')
	print('Training new predictor')
	if not os.path.isfile(model_state):
		latent_model.train_loop(latents_loaders,epochs=31,test_freq=None,vis_freq=None,save_freq=1)
	else:
		latent_model.load_state(model_state)

	tl = latent_model.loss['train']
	x = list(range(len(tl)))

	ax = plt.gca()
	#print(tl)

	ax.plot(x,list(tl.values()))
	ax.set_title('Latent SDE Prediction Error')
	ax.set_xlabel('Training Epoch')
	ax.set_ylabel('Average Prediction Error')
	plt.savefig(os.path.join(root,'latent_nn_real','train_dynamics.png'))
	plt.close('all')

	print('done loading model')
	c1 = Color("#AF2BBF")
	c2 = Color("#5BC8AF")

	# these will both be returned as lists
	print('getting latents')
	ordered_recs,ordered_normie = latent_model.get_latent(latents_pl_loaders['train']) #replace with latents_loaders for all data
	#test = ordered_recs[0]
	print('decoding')
	subset = np.random.choice(int(len(ordered_normie)//3),1,replace=False)

	#real_specs = stitch_and_decode(embedding_VAE,ordered_normie[-50:-20],day=140,\
	#		path=os.path.join(root,'latent_nn_real','testingtesting2'))

	ordered_recs_umapfit = [np.hstack((rec,
						(np.arange(1,len(rec) + 1,1)/len(rec))[:,np.newaxis])) for rec in ordered_recs]
	ordered_normie_umapfit = [np.hstack((normie[:,0:-1],
						(np.arange(1,len(normie) + 1,1)/len(normie))[:,np.newaxis])) for normie in ordered_normie]
	max_len_n = max(map(len,ordered_normie))
	max_len_r = max(map(len,ordered_recs))

	'''
	new_ordered_recs = []
	new_ordered_normie = []
	for rec,norm in zip(ordered_recs,ordered_normie):
		while len(rec) < max_len_r:
			rec = np.vstack((rec, rec[-1,:]))

		while len(norm) < max_len_n:
			norm = np.vstack((norm,norm[-1,:]))

		new_ordered_recs.append(rec)
		new_ordered_normie.append(norm)

	ordered_recs_paths = np.stack(new_ordered_recs)
	ordered_normie_paths = np.stack(new_ordered_normie)
	'''
	ordered_recs2 = np.vstack(ordered_recs)
	ordered_normie2 = np.vstack(ordered_normie)
	#ordered_recs2 = np.vstack()
	ordered_recs_umapfit2 = np.vstack(ordered_recs_umapfit)
	ordered_normie_umapfit2 = np.vstack(ordered_normie_umapfit)
	#print(ordered_normie2.shape)
	#print(ordered_recs2.shape)



	new_lab = np.ones((ordered_recs2.shape[0],))
	orig_lab = np.zeros((ordered_normie2.shape[0],))

	#all_recs = np.vstack((ordered_recs2,ordered_normie2[:,0:-2]))
	subset = np.random.choice(len(ordered_normie),int(len(ordered_normie)//4),replace=False)

	starting_points = np.stack([traj[0,0:-1] for traj in ordered_normie])
	starting_points = starting_points[subset,:]



	all_recs_umapfit = np.vstack((ordered_recs_umapfit2,ordered_normie_umapfit2))
	#all_recs_for_coefs = np.vstack((ordered_recs2,ordered_normie2))
	all_labs = np.hstack((new_lab,orig_lab))


	umap_transform = umap.UMAP(metric='euclidean',n_components=2, n_neighbors=20, min_dist=0.1, \
			random_state=42)
	#umap_transform_filter = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, \
	#		metric='euclidean', random_state=42)
	#latents_filter = umap_transform_filter.fit_transform(all_recs_umapfit)


	lp = []
	latents = []
	latent_inds = []
	all_spec = []
	ind = 0

#	sel = np.random.permutation(len(trainDays))[:17]
#	print('embedding everything')
#	for ind, audd in enumerate(dsb_audio_dirs_all):
#		print('Round ' + str(ind))png
#		segd = dsb_segment_dirs_all[ind]

#		partition = get_window_partition([audd], [segd], 0.23)
#		loaders = get_fixed_window_data_loaders(partition, segment_params)
#		i = 0
#		l =  np.zeros((len(loaders['train'].dataset), embedding_VAE.z_dim))
#		specs = np.zeros((len(loaders['train'].dataset),128,128))
#		for data in loaders['train']:
#			data = data.to(embedding_VAE.device)
#			with torch.no_grad():
#				mu, _, _ = embedding_VAE.encode(data)
#			mu = mu.detach().cpu().numpy()
#			specs[i:i+len(mu),:,:] = data.detach().cpu().numpy()
#			l[i:i+len(mu)] = mu
#			i += len(mu)

#		latents.append(l)
#		latent_inds.append(ind*np.ones((l.shape[0],)))
#		if ind > 61 or ind < 11:
#			lp.append(l)
#			all_spec.append(specs)
#		ind += 1

#	latents = np.vstack(latents)
#	latents = np.vstack([latents,test_l])	subset = np.random.choice(len(all_recs_umapfit),int(len(all_recs_umapfit)/4),replace=False)

#	latent_inds = np.hstack(latent_inds)
#	latent_inds = np.hstack([latent_inds,-1*np.ones(test_l.shape[0],)])
#	all_spec = np.vstack(all_spec)
#	lp = np.vstack(lp)

	#new_spec,new_lat,new_ind,new_e = make_cleaned_plot(all_recs_umapfit,latents_filter, \
	#		all_spec,all_labs,latents_filter)
	print('getting coefs')
	subset = np.random.choice(len(ordered_normie_umapfit2),int(len(ordered_normie_umapfit2)/4),replace=False)

	#umap_recs =  umap_recs/np.max(umap_recs,axis=0)[np.newaxis,:]
	real_v,real_u = latent_model.get_coefs([numpy_to_tensor(ordered_normie_umapfit2[subset,:])])
	real_v,real_u = np.vstack(real_v),np.vstack(real_u)
	noise_size = np.linalg.norm(real_u,axis=1)
	max_noise = np.amax(noise_size)
	min_noise = np.amin(noise_size)
	#print(np.unique(ordered_normie_umapfit2[:,-1]))
	subset = np.random.choice(len(all_recs_umapfit),int(len(all_recs_umapfit)/8),replace=False)

	print('U Mappin bro?')
	umap_transform.fit(all_recs_umapfit[subset,:])
	print('Transforming')
	subset = np.random.choice(len(ordered_normie_umapfit2),int(len(ordered_normie_umapfit2)/4),replace=False)

	print('simsim')
	cmin = Color('#2e1e3b')
	cmax = Color('#8bdab3')

	#val_range = np.linspace(-1,1,500)
	crange = list(cmin.range_to(cmax,len(plotDays)))
	crange = [color.hex_l for color in crange] # list(c1.range_to(c2,len(trainDays)))]
	umap_fig, umap_ax = plt.subplots()
	ii = 0

	for day_ind, d in enumerate(plotDays):

		if ii < 5:
			plotfiles = [fn for fn in fnames_plot if d in fn]

			tl = latents_motif_ds(plotfiles,\
									transform=numpy_to_tensor)
			tldl = DataLoader(tl, batch_size=1, \
				shuffle=False, num_workers=os.cpu_count()-2)
			ll = {'train': tldl, 'test':tldl}
			orll,onll = latent_model.get_latent(ll['train'])
			subset = np.random.choice(len(onll),int(len(onll)//10),replace=False)


			starting_points = np.stack([traj[0,0:-1] for traj in onll])
			starting_points = starting_points[subset,:]

			dummy_list = []
			id = int(d)
			sp = starting_points[0,:][np.newaxis,:]
			picked = onll[subset[0]][:,0:-1]
			picked_r = orll[subset[0]]
			#print(onll[0].shape)
			#picked = np.hstack([picked, (np.arange(1,picked.shape[0] + 1,1)/picked.shape[0])[:,np.newaxis]])
			trajs = latent_model.sim_trajectory(starting_points,day=id,T = 2,stochastic=True)
			trajs1 = latent_model.sim_trajectory(sp,day=id,T = 30,stochastic=True)
			trajs2 = latent_model.sim_trajectory(sp,day=id,T = 30,stochastic=True)
			#print(trajs1.shape)
			print(picked_r.shape)
			pca_trajs1 = np.squeeze(trajs1.detach().cpu().numpy()).T
			pca_fit_mat = np.vstack([picked,picked_r])
			pca_model = PCA().fit(pca_fit_mat)
			pca_rec = pca_model.transform(picked_r)#np.squeeze(trajs1.detach().cpu().numpy()).T)
			pca_rec_traj = pca_model.transform(pca_trajs1)
			for ii in range(16):
				fakefig,fakeax = plt.subplots()
				a = fakeax.plot(list(range(pca_rec_traj.shape[0])),pca_rec_traj[:,ii])
				#b = fakeax.plot(list(range(picked.shape[0])),pca_normie[:,ii])
				#plts.append(a[0])
				#names.append('PC ' + str(ii + 1))

				plt.legend('Simulated Trajectory')
				fakeax.set_xlabel('Time (samples)')
				fakeax.set_ylabel('PC ' + str(ii+1))
				plt.savefig(os.path.join(root, 'latent_nn_real','plots_take2','pca_sim_pc_' + str(ii + 1) + '.svg'))
				plt.close(fakefig)

			for ii in range(16):
				fakefig,fakeax = plt.subplots()
				a = fakeax.plot(list(range(pca_trajs1.shape[0])),pca_trajs1[:,ii])
				#b = fakeax.plot(list(range(picked.shape[0])),pca_normie[:,ii])
				#plts.append(a[0])
				#names.append('PC ' + str(ii + 1))

				plt.legend('Simulated Trajectory')
				fakeax.set_xlabel('Time (samples)')
				fakeax.set_ylabel('Latent Component ' + str(ii+1))
				plt.savefig(os.path.join(root, 'latent_nn_real','plots_take2','sim_latent_component_' + str(ii + 1) + '.svg'))
				plt.close(fakefig)
			#print(pca_rec.shape)
			#print(np.squeeze(trajs1.detach().cpu().numpy()).shape)
			plts = []
			names = []
			pca_normie = pca_model.transform(picked)
			for ii in range(16):
				fakefig,fakeax = plt.subplots()
				a = fakeax.plot(list(range(picked_r.shape[0])),pca_rec[:,ii])
				b = fakeax.plot(list(range(picked.shape[0])),pca_normie[:,ii])
				#plts.append(a[0])
				names.append('PC ' + str(ii + 1))

				plt.legend([a[0],b[0]],['Reconstruction','Real'])
				fakeax.set_xlabel('Time (samples)')
				fakeax.set_ylabel('PC ' + str(ii+1))
				plt.savefig(os.path.join(root, 'latent_nn_real','plots_take2','trying_PCA_rec_PC_' + str(ii + 1) + '.svg'))
				plt.close(fakefig)

			for ii in range(16):
				fakefig,fakeax = plt.subplots()
				a = fakeax.plot(list(range(picked_r.shape[0])),picked_r[:,ii])
				b = fakeax.plot(list(range(picked.shape[0])),picked[:,ii])
				#plts.append(a[0])
				names.append('Latent Component ' + str(ii + 1))

				plt.legend([a[0],b[0]],['Reconstruction','Real'])
				fakeax.set_xlabel('Time (samples)')
				fakeax.set_ylabel('Latent Component ' + str(ii+1))
				plt.savefig(os.path.join(root, 'latent_nn_real','plots_take2','trying_rec_latent_component_' + str(ii + 1) + '.svg'))
				plt.close(fakefig)

			'''
			print('picked shape: ',picked.shape)
			print('pca shape: ', pca_normie.shape)
			#pca_rec = PCA().fit_transform(trajs.detach().cpu().numpy())
			plts = []
			names = []
			fakefig,fakeax = plt.subplots()
			for ii in range(10):

				a = fakeax.plot(list(range(picked.shape[0])),pca_normie[:,ii])
				plts.append(a[0])
				names.append('PC ' + str(ii + 1))

			plt.legend(plts,names)
			fakeax.set_xlabel('Time (samples)')
			fakeax.set_ylabel('PC value')
			plt.savefig(os.path.join(root, 'latent_nn_real','plots_take2','trying_PCA_normie.svg'))
			plt.close(fakefig)
			'''
			umap_trajs = trajs.cpu().detach().numpy()
			trajs1 = trajs1.cpu().detach().numpy()
			trajs2 = trajs2.cpu().detach().numpy()
			false_false = np.squeeze(np.stack([trajs1,trajs2]))
			print(umap_trajs.shape)
			umap_trajs = [np.vstack([t,(np.arange(1,t.shape[1] + 1,1)/t.shape[1])[np.newaxis,:]]) for t in umap_trajs]
			umap_trajs = np.hstack(umap_trajs)
			print(umap_trajs.shape)
			umap_trajs = umap_trajs.T
			#print(umap_trajs.shape)
			uumap_trajs = umap_transform.transform(umap_trajs)


			#ax = plt.gca()
			n = umap_ax.scatter(uumap_trajs[:,0],uumap_trajs[:,1],c=crange[day_ind],s=0.05,alpha=0.2)

			#ax.legend([n,r],['Data','Reconstructions'])

			#ii += 1
			subset = np.random.choice(trajs.shape[0],10,replace=False)
			print('Decoding shape:',false_false.shape)
			sim_specs = stitch_and_decode(embedding_VAE,false_false,day=id,path=os.path.join(root,'latent_nn_real','plots_take2','sims'))
			sim_specs = stitch_and_decode(embedding_VAE,picked[np.newaxis,:],day=id,path=os.path.join(root,'latent_nn_real','plots_take2','sims'))

	umap_ax.set_xlabel('UMAP 1')
	umap_ax.set_ylabel('UMAP 2')
	plt.figure(umap_fig.number)
	plt.savefig(os.path.join(root, 'latent_nn_real','plots_take2','sim_dist_across_d.svg'))
	plt.close('all')

	umap_normie = umap_transform.transform(ordered_normie_umapfit2[subset,:])
	umap_recs = umap_transform.transform(ordered_recs_umapfit2[subset,:])


	ax = plt.gca()
	#r = ax.scatter(umap_recs[:,0],umap_recs[:,1],c='r',s=0.05,alpha=1)
	n = ax.scatter(umap_normie[:,0],umap_normie[:,1],c='b',s=0.05,alpha=0.1)
	ax.set_xlabel('UMAP 1')
	ax.set_ylabel('UMAP 2')
	#ax.legend([n,r],['Data','Reconstructions'])
	plt.savefig(os.path.join(root, 'latent_nn_real','checking_legitness_adult.svg'))
	plt.close('all')

	dvm,dum,dvsd,dusd = [],[],[],[]
	pltax = [int(d) for d in plotDays]

	# indent for real data
	print('getting quivers')
	for d in plotDays:
		train_d = int(d)

		ax = plt.gca()
	#	pts = ax.scatter(umap_normie[:,0],umap_normie[:,1],color=[0.5,0.5,0.5],alpha=0.75,s=0.05)

		plotfiles = [fn for fn in fnames_all if d in fn]
		tl = latents_motif_ds(plotfiles,\
										transform=numpy_to_tensor)
		tldl = DataLoader(tl, batch_size=1, \
				shuffle=False, num_workers=os.cpu_count()-2)
		ll = {'train': tldl, 'test':tldl}
		orll,onll = latent_model.get_latent(ll['train'])
		onll = [np.hstack((n[:,0:-1],
							(np.arange(1,len(n) + 1,1)/len(n))[:,np.newaxis])) for n in onll]
		orll = [np.hstack((r,(np.arange(1,len(r) + 1,1)/len(r))[:,np.newaxis])) for r in orll]


		ss = np.random.choice(len(onll),1,replace=False)
		for s in ss:
			tn = onll[s][:,0:-1].T
			rn = orll[s][:,0:-1].T
			print(tn.shape)
			print(rn.shape)
			sim_specs = stitch_and_decode(embedding_VAE,rn[np.newaxis,:,:],day=s,path=os.path.join(root,'latent_nn_real','plots_take2','pred_recons'))
			sim_specs = stitch_and_decode(embedding_VAE,tn[np.newaxis,:,:],day=s,path=os.path.join(root,'latent_nn_real','plots_take2','recons'))
			st = tn[0,:-1][np.newaxis,:]
			#print(st.shape)
		#	orlln = np.squeeze(latent_model.sim_trajectory(st,train_d,T=50).cpu().detach().numpy())
		#	orlln = orlln.T #day = int(d)
		#	print(orlln.shape)
		#	orlln = np.hstack([orlln, (np.arange(1,len(orlln) + 1,1)/len(orlln))[:,np.newaxis]])
			#### ppppplot here later
		#	uonll = umap_transform.transform(tn)
		#	uorlln = umap_transform.transform(orlln)
		#	ax.plot(uonll[:,0],uonll[:,1],'r')
		#	ax.plot(uorlln[:,0],uorlln[:,1],'b')
		#ax.plot(  ,  , 'b')

		#ax.set_xlabel('UMAP Dim 1')
		#ax.set_ylabel('UMAP Dim 2')
		#plt.savefig(os.path.join(root,'latent_nn_real','plots_take2','aaa' + d + '.svg')) # + d
		#plt.close('all')

		onll = np.vstack(onll)
		orll = np.vstack(orll)
		#umap_onll = umap_transform.transform(onll)
		dv, du = latent_model.get_coefs([numpy_to_tensor(orll)])
		du = np.vstack(du)
		dv = np.vstack(dv)
		#umap_dv = umap_transform.transform(dv)

		sv,su = np.linalg.norm(dv,axis=1),np.linalg.norm(du,axis=1)
		#vals_vec = np.hstack(set_colors(su,min_noise,max_noise))
		dvm.append(np.mean(sv))
		dum.append(np.mean(su))
		dvsd.append(np.std(sv))
		dusd.append(np.std(su))



		#ax = plt.gca()
		#ax.set_title('Quiver of drift: UMAP!')
		#ax.quiver(umap_onll[:,0],umap_onll[:,1],umap_dv[:,0],umap_dv[:,1],vals_vec,angles='xy', scale_units='xy',scale = 10)#,scale=0.05)#,drift_size_umap)#,alpha = 0.5,angles='xy', scale_units='xy', scale=0.01)
		#ax.set_xlabel('UMAP Dim 1')
		#ax.set_ylabel('UMAP Dim 2')
		#ax.set_xlim(0,20)
		#ax.set_ylim(-4,10)
		#plt.savefig(os.path.join(root, 'latent_nn_real','plots_take2','quiver_umap_' + d +'dph.svg')) # + d
		#plt.close('all')

	f,(ax1,ax2) = plt.subplots(1,2,tight_layout=True)
	plt.rcParams['text.usetex'] = True
	yerrv_b, yerrv_t = [di - dvsd[ii] for ii, di in enumerate(dvm)], [di + dvsd[ii] for ii, di in enumerate(dvm)]
	ax1.plot(pltax,dvm)
	ax1.fill_between(pltax,yerrv_b, yerrv_t, color='gray',alpha=0.2)
	ax1.set_xlabel('Days post hatch')
	ax1.set_ylabel(r'$||\mathbf{f}||$')
	yerru_b, yerru_t = [di - dusd[ii] for ii, di in enumerate(dum)], [di + dusd[ii] for ii, di in enumerate(dum)]
	ax2.plot(pltax,dum)
	ax2.fill_between(pltax,yerru_b, yerru_t, color='gray',alpha=0.2)
	ax2.set_ylabel(r'$||\mathbf{g}||$')
	ax2.set_xlabel('Days post hatch')
	plt.savefig(os.path.join(root,'latent_nn_real','plots_take2','terms_size.svg'))
	plt.close('all')



	print('Done plotting!')
	'''
	print('transforming all data coefs')
	#print(real_v.shape)
	#print(ordered_recs_umapfit2[:,-1].shape)
	real_v = np.hstack([real_v, ordered_normie_umapfit2[subset,-1][:,np.newaxis]])
	umap_v = umap_transform.transform(real_v)
	drift_size = np.linalg.norm(real_v,axis=1)
	drift_size_umap = np.linalg.norm(umap_v,axis=1)
	#umap_v = umap_v#/16 #np.max(umap_v,axis=0)[np.newaxis,:]


	ax = plt.gca()
	ax.set_title('Quiver of drift: UMAP!')
	ax.quiver(umap_normie[:,0],umap_normie[:,1],umap_v[:,0],umap_v[:,1],noise_size,angles='xy', scale_units='xy',scale = 10)#,scale=0.05)#,drift_size_umap)#,alpha = 0.5,angles='xy', scale_units='xy', scale=0.01)
	ax.set_xlabel('UMAP Dim 1')
	ax.set_ylabel('UMAP Dim 2')
	#ax.set_xlim(0,20)
	#ax.set_ylim(-4,10)
	plt.savefig(os.path.join(root, 'latent_nn_real','quiver_umap_test.png'))
	#ax.set_xlim(-1001,-730)
	#ax.set_ylim(130,140)
	#plt.savefig(os.path.join(root, 'latent_nn_test','quiver_umap_test_area2.png'))
	#ax.set_xlim(730,1001)
	#ax.set_ylim(130,140)
	#plt.savefig(os.path.join(root, 'latent_nn_test','quiver_umap_test_area3.png'))
	plt.close('all')
	'''
	#print('simsim')
	#trajs = latent_model.sim_trajectory(starting_points,day=1)
	#sim_specs = stitch_and_decode(embedding_VAE,trajs,day=1,path=os.path.join(root,'latent_nn_real','plots_take2'))
	'''
	pca_transform = PCA(n_components=32)
	pca_grid = PCA(n_components=2)

	pca_grid.fit(all_recs)
	pca_transform.fit(all_recs)

	paths_normie = np.percentile(ordered_normie_paths,[25,50,75],axis=0)
	paths_normie_pca = paths_normie[:,:,0:-2] @ pca_grid.components_.T
	dv_medn, ds_medn = latent_model.get_coefs([numpy_to_tensor(paths_normie)])
	paths_recs = np.percentile(ordered_recs_paths,[25,50,75],axis=0)
	paths_recs_pca = paths_recs[:,:,0:-2] @ pca_grid.components_.T
	dv_medr, ds_medr = latent_model.get_coefs([numpy_to_tensor(paths_recs)])
	#dv_medn = np.stack(dv_medn)
	print(dv_medn.shape)
	ax = plt.gca()
	ax.plot(paths_normie_pca[0,:,0],paths_normie_pca[0,:,1],c='r')
	ax.plot(paths_normie_pca[1,:,0],paths_normie_pca[1,:,1],'--r')
	ax.plot(paths_normie_pca[2,:,0],paths_normie_pca[2,:,1],'--r')

	ax.plot(paths_recs_pca[0,:,0],paths_recs_pca[0,:,1],c='b')
	ax.plot(paths_recs_pca[1,:,0],paths_recs_pca[1,:,1],'--b')
	ax.plot(paths_recs_pca[2,:,0],paths_recs_pca[2,:,1],'--b')

	plt.savefig(os.path.join(root, 'latent_nn_test','median_trajectories_pca.png'))
	spacing = np.linspace(-2,2,25)

	positx,posity = np.meshgrid(spacing,spacing)
	plotx,ploty,plotu01,plotv01,plot_col01 = [],[],[],[],[]
	plotu23,plotv23,plot_col23 = [],[],[]

	plotu01o,plotv01o,plot_col01o = [],[],[]
	plotu23o,plotv23o,plot_col23o = [],[],[]
	for x,y in zip(positx,posity):
		d2 = np.stack((x,y)).T
		d32 = d2 @ pca_grid.components_
		#print(d32.shape)

		d34 = np.hstack([d32,np.ones((d32.shape[0],1)),np.ones((d32.shape[0],1))])
		#print(len([d34]))
		dv,ds = latent_model.get_coefs([numpy_to_tensor(d34)])

		#print(dv.shape)
		#print(ds.shape)
		dv_pc = dv[:,0:-2] @ pca_transform.components_
		norm_dv = np.max(np.linalg.norm(dv,axis=1))
		norm_dvpc = np.max(np.linalg.norm(dv_pc,axis=1))
		ds_pc = ds[:,0:-2] @ pca_transform.components_

		plotx.append(x)
		ploty.append(y)
		plotu01.append(dv_pc[:,0]/norm_dvpc)
		plotv01.append(dv_pc[:,1]/norm_dvpc)
		plot_col01.append(np.linalg.norm(ds_pc[:,0:1],axis=1))
		plotu23.append(dv_pc[:,2]/norm_dvpc)
		plotv23.append(dv_pc[:,3]/norm_dvpc)
		plot_col23.append(np.linalg.norm(ds_pc[:,2:3],axis=1))

		plotu01o.append(dv[:,0]/norm_dv)
		plotv01o.append(dv[:,1]/norm_dv)
		plot_col01o.append(np.linalg.norm(ds[:,0:1],axis=1))
		plotu23o.append(dv[:,2]/norm_dv)
		plotv23o.append(dv[:,3]/norm_dv)
		plot_col23o.append(np.linalg.norm(ds[:,2:3],axis=1))
		#plot_col2.append(ds_pc[:,1])

	#print(dv.shape)
	#print(ds_pc.shape)
	plotx = np.stack(plotx)
	ploty = np.stack(ploty)
	plotu01 = np.stack(plotu01)
	plotv01 = np.stack(plotv01)
	cols01 = np.stack(plot_col01)
	plotu23 = np.stack(plotu23)
	plotv23 = np.stack(plotv23)
	cols23 = np.stack(plot_col23)

	plotu01o = np.stack(plotu01o)
	plotv01o = np.stack(plotv01o)
	cols01o = np.stack(plot_col01o)
	plotu23o = np.stack(plotu23o)
	plotv23o = np.stack(plotv23o)
	cols23o = np.stack(plot_col23o)


	ax = plt.gca()
	ax.set_title('Quiver of drift in first two PCs')
	ax.quiver(plotx,ploty,plotu01,plotv01,cols01)
	ax.set_xlabel('PC 1')
	ax.set_ylabel('PC 2')
	plt.savefig(os.path.join(root, 'latent_nn_test','quiver_test_01.png'))
	plt.close('all')

	ax = plt.gca()
	ax.set_title('Quiver of drift in third and fourth PCs')
	ax.quiver(plotx,ploty,plotu23,plotv23,cols23)
	ax.set_xlabel('PC 3')
	ax.set_ylabel('PC 4')
	plt.savefig(os.path.join(root, 'latent_nn_test','quiver_test_23.png'))
	plt.close('all')

	ax = plt.gca()
	ax.set_title('Quiver of drift in first two latent dims')
	ax.quiver(plotx,ploty,plotu01o,plotv01o,cols01o)
	#ax.quiver(paths_normie_pca[:,:,0],paths_normie_pca[:,:,1],dv_medn[:,0],dv_medn[:,1])

	ax.set_xlabel('D 1')
	ax.set_ylabel('D 2')
	plt.savefig(os.path.join(root, 'latent_nn_test','quiver_test_01o.png'))
	plt.close('all')

	ax = plt.gca()
	ax.set_title('Quiver of drift in third and fourth latent dims')
	ax.quiver(plotx,ploty,plotu23o,plotv23o,cols23o)
	#ax.quiver(paths_normie_pca[:,:,0],paths_normie_pca[:,:,1],dv_medn[:,2],dv_medn[:,3])

	ax.set_xlabel('D 3')
	ax.set_ylabel('D 4')
	plt.savefig(os.path.join(root, 'latent_nn_test','quiver_test_23o.png'))
	plt.close('all')
	'''

	'''
	umap_new_lats = umap_transform.fit_transform(all_recs)

	c1r = Color("#F7A08D")
	c2r = Color("#941D04")

	c1b = Color("#6063F1")
	c2b = Color("#0C12D8")
	for ind, rec in enumerate(ordered_recs):
		if np.random.rand() > 0.5:
			r = rec[:,0:-2]
			n = ordered_normie[ind][:,0:-2]

			cranger = [color.hex_l for color in list(c1r.range_to(c2r,r.shape[0]))]
			crangeb = [color.hex_l for color in list(c1b.range_to(c2b,r.shape[0]))]

			r_umap = umap_transform.transform(r)
			n_umap = umap_transform.transform(n)

			ax = plt.gca()
			ax.plot(r_umap[:,0],r_umap[:,1],c=cranger[0])
			ax.plot(n_umap[:,0],n_umap[:,1],c=crangeb[0])
			plt.savefig(os.path.join(root,'latent_nn_test','trying_out_latents' + str(ind) + '.png'))
			plt.close('all')



	crange = [color.hex_l for color in list(c1.range_to(c2,len(trainDays)))]

	test_l = embedding_VAE.get_latent(loaders['train'])
	umap_transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, \
	metric='euclidean', random_state=42)

	umap_latents = umap_transform.fit_transform(test_l)
	ax = plt.gca()
	ax.scatter(umap_latents[:,0],umap_latents[:,1])
	plt.savefig(os.path.join(root,'testfig.png'))
	plt.close('all')

	lp = []
	latents = []
	latent_inds = []
	all_spec = []
	ind = 0

	sel = np.random.permutation(len(trainDays))[:17]
	print('embedding everything')
	for ind, audd in enumerate(dsb_audio_dirs_all):
		print('Round ' + str(ind))
		segd = dsb_segment_dirs_all[ind]

		partition = get_window_partition([audd], [segd], 0.23)
		loaders = get_fixed_window_data_loaders(partition, segment_params)
		i = 0
		l =  np.zeros((len(loaders['train'].dataset), embedding_VAE.z_dim))
		specs = np.zeros((len(loaders['train'].dataset),128,128))
		for data in loaders['train']:
			data = data.to(embedding_VAE.device)
			with torch.no_grad():
				mu, _, _ = embedding_VAE.encode(data)
			mu = mu.detach().cpu().numpy()
			specs[i:i+len(mu),:,:] = data.detach().cpu().numpy()
			l[i:i+len(mu)] = mu
			i += len(mu)

		latents.append(l)
		latent_inds.append(ind*np.ones((l.shape[0],)))
		if ind > 61 or ind < 11:
			lp.append(l)
			all_spec.append(specs)
		ind += 1

	latents = np.vstack(latents)
	latents = np.vstack([latents,test_l])
	latent_inds = np.hstack(latent_inds)
	latent_inds = np.hstack([latent_inds,-1*np.ones(test_l.shape[0],)])
	all_spec = np.vstack(all_spec)
	lp = np.vstack(lp)
	print(latents.shape)
	print(all_spec.shape)
	print('running UMAP')
	umap_transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, \
	metric='euclidean', random_state=42)
	umap_latents = umap_transform.fit_transform(latents)
	ep = umap_transform.transform(lp)
	new_spec,new_lat,new_ind,new_e = make_cleaned_plot(lp,ep,all_spec,latent_inds,umap_latents)


	new_ind = new_ind.astype(int)
	fns = []

	xy_max = np.amax(new_e,axis=0)
	xy_min = np.amin(new_e,axis=0)

	cc = []
	cd = []

	n = 30
	#e_t = umap_transform.transform(test_l)

	for ii in range(ind):

		print(ii)

		cd.append(new_e[new_ind == ii, :])
		cc.append(crange[ii])
		if len(cd) > n:
			cd.pop(0)
			cc.pop(0)

		ax = plt.gca()
		for jj, e in enumerate(cd):
			ax.scatter(e[:,0],e[:,1],c=cc[jj],s=0.05,alpha=1/((0.2*jj)+1))
		#ax.scatter(new_e[new_ind == -1,0],new_e[new_ind==-1,1],c='r',s=0.1,alpha=1)
		#ax.set_title('Day ' + trainDays[ii], loc='left')
		#ax.scatter(new_e[new_ind == ii ,0],new_e[new_ind == ii,1],c=crange[ii],s=0.05)
		#plt.tight_layout()
		#plt.axis('square')
		plt.axis('off')

		ax.set_xlim(xy_min[0] - 3, xy_max[0] + 3)
		ax.set_ylim(xy_min[1] - 3, xy_max[1] + 3)
		fn = os.path.join(root,'movie_im' + str(ii) + '.png')
		plt.savefig(fn)
		#plt.close('all')
		fns.append(fn)
		plt.close('all')

	images = []
	for fn in fns:
		images.append(imageio.imread(fn))

	imageio.mimsave(os.path.join(root,'latent_movie.gif'),images)
	'''
	return

@numba.njit(fastmath=True)
def time_weighted_distance(x,y,c = 10):
	"""
	euclidean distance, weighted by time

	..math::
		D(x,y) = \exp{c * |t_m - t_n|}\sqrt{\sum_i (x_i - y_i)^2}
	"""

	tm = x[-1]
	tn = y[-1]

	x = x[0:-1]
	y = y[0:-1]

	#a = abs(x - y)
	distance_sqr = 0.0
	weight= np.exp(c * abs(tm - tn))#/(1 + np.exp(c * abs(tm - tn)))

	#if abs(tm - tn) <= 1:
		#eeweight = c
	#else:
	#	weight = 1

	g = np.zeros_like(x)

	for ii in range(x.shape[0]):
		a = abs(x[ii] - y[ii])
		distance_sqr += a ** 2
		g[ii] = weight * (x[ii] - y[ii])

	distance = weight * np.sqrt(distance_sqr)
	#print(distance)
	#print(g)
	return distance, g/(1e-6 + distance)

if __name__ == '__main__':

	root = '/home/mrmiews/Desktop/Pearson_Lab/'
	fire.Fire(bird_model_script) #(n_train_runs = 1, root = os.path.join(root,'model_for_movie'),figs=False)
