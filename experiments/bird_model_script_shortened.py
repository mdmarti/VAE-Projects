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
from sklearn.decomposition import PCA
import umap
import numpy as np
import os


from ava.preprocessing.utils import get_spec
from ava.segmenting.segment import tune_segmenting_params
from ava.segmenting.amplitude_segmentation import get_onsets_offsets

from ava.models.window_vae_dataset import get_window_partition, \
				get_fixed_ordered_data_loaders_motif

FS = 42000
from ava.models.vae import X_SHAPE

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

	mean_recon_day = mean_recon_day.detach().cpu().numpy()#.astype('uint8')
	
	frames = []
	for spec in mean_recon_day:
		
		frames.append(np.squeeze(spec))

	frames = tuple(frames)
	stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
	stitcher.setPanoConfidenceThresh(0.02)
	(status,stitched) = stitcher.stitch(frames)
	print(status)
	print(type(stitched))
	ax = plt.gca()
	im = ax.imshow(stitched,origin='lower',vmin=0,vmax=1)

	plt.savefig(os.path.join(model.save_dir,'mean_image_.png'))

	return mean_traj, stitched

def model_comparison_umap(vanilla,smoothprior,time_recon,loader,n_samples = 5,day_name='',joint_umap = None,return_umap=True):

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

	if joint_umap is None:
		joint_umap = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, random_state=42)

		print('fittin umap')
		stacked_transformed = joint_umap.fit_transform(latents_stacked)
	else:
		stacked_transformed = joint_umap.transform(latents_stacked)

	vanilla_transformed = joint_umap.transform(stacked_vanilla)
	smooth_transformed = joint_umap.transform(stacked_smooth)
	time_transformed = joint_umap.transform(stacked_time)
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
	fg = ax.scatter(vanilla_transformed[:,0],vanilla_transformed[:,1],s=0.25,alpha=0.05,color='#FAC559')
	for s in vanilla_samps:
		tmp_samp = latents_vanilla[s]
		tmp_umap = joint_umap.transform(tmp_samp)
		#ax.plot(tmp_umap[:,0],tmp_umap[:,1], color='r')
	if day_name == '':
		plt.savefig(os.path.join(vanilla.save_dir,'vanilla_latent_samples.png'))
	else:
		plt.savefig(os.path.join(vanilla.save_dir,'vanilla_latent_samples_' + day_name + '.png'))
	plt.close('all')
	ax = plt.gca()
	print('plotting smooth prior latents')
	bg = ax.scatter(stacked_transformed[:,0],stacked_transformed[:,1],s=0.25,alpha=0.05,color='k')
	fg = ax.scatter(smooth_transformed[:,0],smooth_transformed[:,1],s=0.25,alpha=0.05,color='#E0B5D3')
	for s in smooth_samps:
		tmp_samp = latents_smoothprior[s]
		tmp_umap = joint_umap.transform(tmp_samp)
		#ax.plot(tmp_umap[:,0],tmp_umap[:,1], color='r')

	if day_name == '':
		plt.savefig(os.path.join(smoothprior.save_dir,'smoothprior_latent_samples.png'))
	else:
		plt.savefig(os.path.join(smoothprior.save_dir,'smoothprior_latent_samples_' + day_name + '.png'))
	plt.close('all')
	ax = plt.gca()
	print('plotting time recon latents')
	bg = ax.scatter(stacked_transformed[:,0],stacked_transformed[:,1],s=0.25,alpha=0.05,color='k')
	fg = ax.scatter(time_transformed[:,0],time_transformed[:,1],s=0.25,alpha=0.05,color='#68A9CF')
	for s in time_samps:
		tmp_samp = latents_time[s]
		tmp_umap = joint_umap.transform(tmp_samp)
		#ax.plot(tmp_umap[:,0],tmp_umap[:,1], color='r')
	if day_name == '':
		plt.savefig(os.path.join(time_recon.save_dir,'timerecon_latent_samples.png'))
	else:
		plt.savefig(os.path.join(time_recon.save_dir,'timerecon_latent_samples_' + day_name + '.png'))
	plt.close('all')

	if return_umap:
		return joint_umap
	else:
		return





def bird_model_script(vanilla_dir='',smoothness_dir = '',time_recondir = '',datadir='',segment=False):

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
		'window_overlap':0.05, # overlap between spec windows, in s
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
	if segment:
		segment_params = tune_segmenting_params(dsb_audio_dirs,segment_params)

		from ava.segmenting.segment import segment
		for audio_dir, segment_dir in zip(dsb_audio_dirs, dsb_segment_dirs):
			segment(audio_dir, segment_dir, segment_params)


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
		vanilla_vae = VAE_Base(vanilla_encoder,vanilla_decoder,vanilla_dir,plots_dir=os.path.join(vanilla_dir,'plots_shortwindow'))

		if not os.path.isfile(save_file):
			print('training vanilla')
			vanilla_vae.train_test_loop(loaders_for_prediction,epochs=301,test_freq=5,save_freq=50,vis_freq=25)
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
			#print(loaders[0])
			#mean,_ = smoothness_analysis(vanilla_vae,loaders[0]['train'])

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
		smooth_prior_vae = SmoothnessPriorVae(smooth_encoder,smooth_decoder,smoothness_dir,plots_dir=os.path.join(smoothness_dir,'plots_shortwindow'))

		if not os.path.isfile(save_file):
			print('training smooth')
			smooth_prior_vae.train_test_loop(loaders_for_prediction,epochs=301,test_freq=5,save_freq=50,vis_freq=25)
		else:
			print('loading smooth')
			smooth_prior_vae.load_state(save_file)
			#smooth_prior_vae.train_test_loop(loaders_for_prediction,epochs=151,test_freq=5,save_freq=50,vis_freq=25)
			#smooth_prior_vae.test_epoch(loaders_for_prediction['test'])
			#mean,_ = smoothness_analysis(smooth_prior_vae,loaders[0]['train'])
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
		time_vae = ReconstructTimeVae(time_encoder,time_decoder,time_recondir,plots_dir=os.path.join(time_recondir,'plots_shortwindow'))

		if not os.path.isfile(save_file):
			print('training time')
			time_vae.train_test_loop(loaders_for_prediction,epochs=301,test_freq=5,save_freq=50,vis_freq=25)
		else:
			print('loading time')
			time_vae.load_state(save_file)
			#time_vae.test_epoch(loaders_for_prediction['test'])
			#time_vae.train_test_loop(loaders_for_prediction,epochs=151,test_freq=5,save_freq=50,vis_freq=25)
			#mean,_ = smoothness_analysis(time_vae,loaders[0]['train'])

			'''
			for ind, l in enumerate(loaders):
				print('Developmental day {} \n'.format(realTrainDays[ind]))
				_,_ = pca_analysis(time_vae,l['train'])
			'''

	print('doing model comparison')
	joint_umap = model_comparison_umap(vanilla_vae,smooth_prior_vae,time_vae,loaders_for_prediction['test'],day_name='')

	for day in trainDays:
		motif_part = get_window_partition([os.path.join(datadir,dsb[0],'audio',day)],[os.path.join(datadir,dsb[0],'syll_segs',day)],1.0)
		motif_part['test'] = motif_part['train']
		print('getting prediction loader')
		loaders_for_prediction = get_fixed_ordered_data_loaders_motif(motif_part,segment_params)
		model_comparison_umap(vanilla_vae,smooth_prior_vae,time_vae,loaders_for_prediction['test'],day_name=day,joint_umap=joint_umap,return_umap=False)

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


if __name__ == '__main__':

	root = '/home/mrmiews/Desktop/Pearson_Lab/'
	fire.Fire(bird_model_script) #(n_train_runs = 1, root = os.path.join(root,'model_for_movie'),figs=False)
