
import os
import sys
import inspect


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir) 

import fire
from VAE_Projects.models.utils import get_window_partition,get_simsiam_loaders_motif,batch_cos_sim,batch_mse
from VAE_Projects.models.simsiam_models import encoder,predictor,simsiam,simsiam_l1,simsiam_l2,normed_simsiam
from VAE_Projects.experiments.simsiam_analysis import lookin_at_latents,z_plots
from VAE_Projects.experiments.plotting import plot_trajectories_umap_and_coords
import matplotlib.pyplot as plt
from colour import Color
import numpy as np
import os
import cv2
from sklearn.decomposition import PCA
import umap
import numpy as np
import os
import seaborn as sns
from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader


from ava.preprocessing.utils import get_spec
from ava.segmenting.segment import tune_segmenting_params
from ava.segmenting.amplitude_segmentation import get_onsets_offsets

from ava.models.window_vae_dataset import get_window_partition, \
				get_fixed_ordered_data_loaders_motif

FS = 42000
from ava.models.vae import X_SHAPE

def imagenet_script(imagenetdir = '',simsiam_dir='',batch_size=128,shuffle=(True,False),num_workers=4):

	ts=transforms.Compose([transforms.RandomApply(transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.1),p=0.8),\
				transforms.RandomResizedCrop(size=(32,32),scale=(0.2,1.0)),\
				transforms.RandomHorizontalFlip(),\
				transforms.RandomGrayscale(p=0.2),\
				transforms.GaussianBlur(kernel_size=3,sigma=(0.1,2.0))])
	
	ds_train = ImageNet(root=os.path.join(imagenetdir,'train'),split='train')
	ds_val = ImageNet(root=os.path.join(imagenetdir,'val'),split='val')

	
	dls = {'train':DataLoader(ds_train,batch_size=batch_size, \
			shuffle=shuffle[0], num_workers=num_workers),\
			'test':DataLoader(ds_train,batch_size=batch_size, \
			shuffle=shuffle[1], num_workers=num_workers)}


	if simsiam_dir != '':
		if not os.path.isdir(simsiam_dir):
			os.mkdir(simsiam_dir)
		save_file = os.path.join(simsiam_dir,'checkpoint_encoder_300.tar')
		#print(save_file)
		simsiam_encoder = encoder()
		simsiam_predictor = predictor()
		vanilla_simsiam = simsiam(simsiam_encoder,simsiam_predictor,save_dir=simsiam_dir,sim_func=batch_cos_sim)

		if not os.path.isfile(save_file):
			print('training vanilla')
			vanilla_simsiam.train_test_loop(dls,epochs=301,test_freq=5,save_freq=50)
			train_latents = vanilla_simsiam.get_latent(dls['train'])
			test_latents = vanilla_simsiam.get_latent(dls['test'])
			print('here')
			lookin_at_latents(vanilla_simsiam,dls['train'])

			l_umap = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, random_state=42)

			train_umap = l_umap.fit_transform(np.vstack(train_latents))
			test_umap = l_umap.transform(np.vstack(test_latents))


			ax = plt.gca()
			sns.scatterplot(x=train_umap[:,0],y=train_umap[:,1],markers='+',ax=ax)
			sns.scatterplot(x=test_umap[:,0],y=test_umap[:,1],markers='o',ax=ax)

			plt.savefig(os.path.join(simsiam_dir,'latents.png'))
			plt.close('all')
		else:
			print('loading vanilla')
			vanilla_simsiam.load_state(save_file)
			train_latents = vanilla_simsiam.get_latent(dls['train'])
			test_latents = vanilla_simsiam.get_latent(dls['test'])
			lookin_at_latents(vanilla_simsiam,dls['train'])

			l_umap = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, random_state=42)

			train_umap = l_umap.fit_transform(np.vstack(train_latents))
			test_umap = l_umap.transform(np.vstack(test_latents))


			ax = plt.gca()
			sns.scatterplot(x=train_umap[:,0],y=train_umap[:,1],markers='+',ax=ax)
			sns.scatterplot(x=test_umap[:,0],y=test_umap[:,1],markers='o',ax=ax)

			plt.savefig(os.path.join(simsiam_dir,'latents.png'))
			plt.close('all')

	return

def bird_model_script(simsiam_dir='',simsiam_l1_dir='',simsiam_masked_dir='',simsiam_l2_dir='',normed=False,segment=False,wd=False):

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
	adult_audio_dirs = ['/hdd/miles/birds/blk417_tutor/motif_audio_tutor',
						'/hdd/miles/birds/blk411_tutor/motif_audio_tutor']
	adult_motif_dirs = ['/hdd/miles/birds/blk417_tutor/motif_segs',
						'/hdd/miles/birds/blk411_tutor/motif_segs']


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
	# now trained on window length 0.12, overlap 0.06
	# short window is length 0.10, overlap 0.05
	# shorter is length 0.08, overlap 0.04
	#shortshorter is 0.06,overlap 0.03
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
		'window_length': 0.12, # spec window, in s
		'window_overlap':0.11, # overlap between spec windows, in s
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
        'train_augment': True, # whether or not we are training simsiam
		'max_tau':0.01 # max horizontal time shift for our window
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
	#motif_part = get_window_partition(adult_audio_dirs,adult_motif_dirs,0.8)
	#motif_part['test'] = motif_part['train']
	print('getting prediction loader')
	#loaders_for_prediction = get_simsiam_loaders_motif(motif_part,segment_params,n_samples=2000,batch_size=128)
	# this is used for the shotgun VAE, as opposed to the shotgun-dynamics VAE
	partition = get_window_partition(dsb_audio_dirs, dsb_segment_dirs, 0.8)
	loaders = get_simsiam_loaders_motif(partition, segment_params)

#############################
# 1) Train model            #
#############################
	if simsiam_dir != '':
		if not os.path.isdir(simsiam_dir):
			os.mkdir(simsiam_dir)
		save_file = os.path.join(simsiam_dir,'checkpoint_encoder_300.tar')
		#print(save_file)
		simsiam_encoder = encoder(z_dim=64)
		simsiam_predictor = predictor(z_dim=64,h_dim=32)
		if normed:
			vanilla_simsiam = normed_simsiam(simsiam_encoder,simsiam_predictor,save_dir=simsiam_dir,sim_func=batch_cos_sim,wd=wd)
		else:
			vanilla_simsiam = simsiam(simsiam_encoder,simsiam_predictor,save_dir=batch_cos_sim,sim_func=batch_mse,wd=wd)
		if not os.path.isfile(save_file):
			print('training vanilla')
			vanilla_simsiam.train_test_loop(loaders_for_prediction,epochs=301,test_freq=5,save_freq=50)
			lookin_at_latents(vanilla_simsiam,loaders_for_prediction['train'])
		else:
			print('loading vanilla')
			vanilla_simsiam.load_state(save_file)
			#train_latents = vanilla_simsiam.get_latent(loaders_for_prediction['train'])
			#print(len(train_latents))
			#test_latents = vanilla_simsiam.get_latent(loaders_for_prediction['test'])
			#print(len(test))
			#lookin_at_latents(vanilla_simsiam,loaders_for_prediction['test'])
			plot_trajectories_umap_and_coords(vanilla_simsiam,loaders_for_prediction['test'])

			z_plots(vanilla_simsiam,loaders_for_prediction['test'])
			#l_umap = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, random_state=42)

			#train_umap = l_umap.fit_transform(np.vstack(train_latents))
			#test_umap = l_umap.transform(np.vstack(test_latents))


			#ax = plt.gca()
			#sns.scatterplot(x=train_umap[:,0],y=train_umap[:,1],markers='+',ax=ax)
			#sns.scatterplot(x=test_umap[:,0],y=test_umap[:,1],markers='o',ax=ax)

			#plt.savefig(os.path.join(simsiam_dir,'latents.png'))
			#plt.close('all')
			#vanilla_vae.train_test_loop(loaders_for_prediction,epochs=151,test_freq=5,save_freq=50,vis_freq=25)
			#vanilla_vae.test_epoch(loaders_for_prediction['test'])

	if simsiam_l1_dir != '':
		if not os.path.isdir(simsiam_l1_dir):
			os.mkdir(simsiam_l1_dir)
		save_file = os.path.join(simsiam_l1_dir,'checkpoint_encoder_300.tar')
		#print(save_file)
		simsiam_l1_encoder = encoder(z_dim=128)
		simsiam_l1_predictor = predictor(z_dim=128,h_dim=64)
		l1_simsiam = simsiam_l1(simsiam_l1_encoder,simsiam_l1_predictor,save_dir=simsiam_l1_dir,sim_func=batch_cos_sim,lamb=1e-12)

		if not os.path.isfile(save_file):
			print('training vanilla')
			l1_simsiam.train_test_loop(loaders_for_prediction,epochs=5001,test_freq=5,save_freq=25)
			lookin_at_latents(l1_simsiam,loaders_for_prediction['train'])
			#lookin_at_latents(l1_simsiam,loaders_for_prediction['test'])
			z_plots(l1_simsiam,loaders_for_prediction['train'])
		else:
			print('loading vanilla')
			l1_simsiam.load_state(save_file)
			train_latents = l1_simsiam.get_latent(loaders_for_prediction['train'])
			#print(len(train_latents))
			test_latents = l1_simsiam.get_latent(loaders_for_prediction['test'])
			#print(len(test))
			lookin_at_latents(l1_simsiam,loaders_for_prediction['train'])
			z_plots(l1_simsiam,loaders_for_prediction['train'])

			l_umap = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, random_state=42)

			train_umap = l_umap.fit_transform(np.vstack(train_latents))
			test_umap = l_umap.transform(np.vstack(test_latents))


			ax = plt.gca()
			sns.scatterplot(x=train_umap[:,0],y=train_umap[:,1],markers='+',ax=ax)
			sns.scatterplot(x=test_umap[:,0],y=test_umap[:,1],markers='o',ax=ax)

			plt.savefig(os.path.join(simsiam_l1_dir,'latents.png'))
			plt.close('all')
			#vanilla_vae.train_test_loop(loaders_for_prediction,epochs=151,test_freq=5,save_freq=50,vis_freq=25)
			#vanilla_vae.test_epoch(loaders_for_prediction['test'])

	if simsiam_masked_dir != '':
		if not os.path.isdir(simsiam_masked_dir):
			os.mkdir(simsiam_masked_dir)
		save_file = os.path.join(simsiam_masked_dir,'checkpoint_encoder_300.tar')
		#print(save_file)
		simsiam_mask_encoder = encoder(z_dim=128)
		simsiam_mask_predictor = predictor(z_dim=128,h_dim=64)
		mask_simsiam = simsiam_l1(simsiam_mask_encoder,simsiam_mask_predictor,save_dir=simsiam_masked_dir,sim_func=batch_cos_sim,lamb=1e-12)

		if not os.path.isfile(save_file):
			print('training vanilla')
			mask_simsiam.train_test_loop(loaders_for_prediction,epochs=5001,test_freq=5,save_freq=25)
			lookin_at_latents(mask_simsiam,loaders_for_prediction['train'])
			#lookin_at_latents(l1_simsiam,loaders_for_prediction['test'])
			z_plots(mask_simsiam,loaders_for_prediction['train'])
		else:
			print('loading vanilla')
			mask_simsiam.load_state(save_file)
			train_latents = mask_simsiam.get_latent(loaders_for_prediction['train'])
			#print(len(train_latents))
			test_latents = mask_simsiam.get_latent(loaders_for_prediction['test'])
			#print(len(test))
			lookin_at_latents(mask_simsiam,loaders_for_prediction['train'])
			z_plots(mask_simsiam,loaders_for_prediction['train'])

			l_umap = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, random_state=42)

			train_umap = l_umap.fit_transform(np.vstack(train_latents))
			test_umap = l_umap.transform(np.vstack(test_latents))


			ax = plt.gca()
			sns.scatterplot(x=train_umap[:,0],y=train_umap[:,1],markers='+',ax=ax)
			sns.scatterplot(x=test_umap[:,0],y=test_umap[:,1],markers='o',ax=ax)

			plt.savefig(os.path.join(simsiam_masked_dir,'latents.png'))
			plt.close('all')
			#vanilla_vae.train_test_loop(loaders_for_prediction,epochs=151,test_freq=5,save_freq=50,vis_freq=25)
			#vanilla_vae.test_epoch(loaders_for_prediction['test'])

	if simsiam_l2_dir != '':
		if not os.path.isdir(simsiam_l2_dir):
			os.mkdir(simsiam_l2_dir)
		save_file = os.path.join(simsiam_l2_dir,'checkpoint_encoder_300.tar')
		#print(save_file)
		simsiam_l2_encoder = encoder(z_dim=128)
		simsiam_l2_predictor = predictor(z_dim=128,h_dim=64)
		l2_simsiam = simsiam_l2(simsiam_l2_encoder,simsiam_l2_predictor,save_dir=simsiam_l2_dir,sim_func=batch_cos_sim,lamb=1e-12)

		if not os.path.isfile(save_file):
			print('training vanilla')
			l2_simsiam.train_test_loop(loaders_for_prediction,epochs=5001,test_freq=5,save_freq=25)
			lookin_at_latents(l2_simsiam,loaders_for_prediction['train'])
			#lookin_at_latents(l1_simsiam,loaders_for_prediction['test'])
			z_plots(l2_simsiam,loaders_for_prediction['train'])
		else:
			print('loading vanilla')
			l2_simsiam.load_state(save_file)
			train_latents = l2_simsiam.get_latent(loaders_for_prediction['train'])
			#print(len(train_latents))
			test_latents = l2_simsiam.get_latent(loaders_for_prediction['test'])
			#print(len(test))
			lookin_at_latents(l2_simsiam,loaders_for_prediction['train'])
			z_plots(l2_simsiam,loaders_for_prediction['train'])

			l_umap = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, random_state=42)

			train_umap = l_umap.fit_transform(np.vstack(train_latents))
			test_umap = l_umap.transform(np.vstack(test_latents))


			ax = plt.gca()
			sns.scatterplot(x=train_umap[:,0],y=train_umap[:,1],markers='+',ax=ax)
			sns.scatterplot(x=test_umap[:,0],y=test_umap[:,1],markers='o',ax=ax)

			plt.savefig(os.path.join(simsiam_l2_dir,'latents.png'))
			plt.close('all')
			'''
			for ind, l in enumerate(loaders):
				print('Developmental day {} \n'.format(realTrainDays[ind]))
				_,_ = pca_analysis(vanilla_vae,l['train'])
			'''

			'''
			
			Add in new analyses here!!!!
			'''
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
	#fire.Fire(imagenet_script)
