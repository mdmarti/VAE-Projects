import numpy as np
import torch
import os 
from colour import Color
import pickle 
import glob
import math
from torch.utils.data import Dataset, DataLoader
import warnings
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy import signal
from scipy.io.wavfile import WavFileWarning
import random
import matplotlib.pyplot as plt

X_SHAPE = (128,128)

############## math ###################################

def batch_cos_sim(x,y):

	if len(x.shape) < 3:
		x = x[:,:,None]
	if len(y.shape) < 3:
		y = y[:,:,None]
	
	norm_x = torch.linalg.norm(x,dim=1,keepdim=True)
	norm_y = torch.linalg.norm(y,dim=1,keepdim=True)

	cs = ((x/norm_x) @ (y/norm_y).transpose(1,2)).squeeze()

	return -cs.mean()


############## dataloaders #############################
def _get_wavs_from_dir(dir):
	"""Return a sorted list of wave files from a directory."""
	return [os.path.join(dir, f) for f in sorted(os.listdir(dir)) if \
			_is_wav_file(f)]

def _is_wav_file(filename):
	"""Is the given filename a wave file?"""
	return len(filename) > 4 and filename[-4:] == '.wav'


def get_window_partition(audio_dirs, roi_dirs, split=0.8, shuffle=True, \
	exclude_empty_roi_files=True):
	"""
	Get a train/test split for fixed-duration shotgun VAE.

	Parameters
	----------
	audio_dirs : list of str
		Audio directories.
	roi_dirs : list of str
		ROI (segment) directories.
	split : float, optional
		Train/test split. Defaults to ``0.8``, indicating an 80/20 train/test
		split.
	shuffle : bool, optional
		Whether to shuffle at the audio file level. Defaults to ``True``.
	exclude_empty_roi_files : bool, optional
		Defaults to ``True``.

	Returns
	-------
	partition : dict
		Defines the test/train split. The keys ``'test'`` and ``'train'`` each
		map to a dictionary with keys ``'audio'`` and ``'rois'``, which both
		map to numpy arrays containing filenames.
	"""
	assert(split > 0.0 and split <= 1.0)
	# Collect filenames.
	audio_filenames, roi_filenames = [], []
	for audio_dir, roi_dir in zip(audio_dirs, roi_dirs):
		temp_wavs = _get_wavs_from_dir(audio_dir)
		temp_rois = [os.path.join(roi_dir, os.path.split(i)[-1][:-4]+'.txt') \
				for i in temp_wavs]
		if exclude_empty_roi_files:
			for i in reversed(range(len(temp_wavs))):
				segs = np.loadtxt(temp_rois[i])
				if len(segs) == 0:
					del temp_wavs[i]
					del temp_rois[i]
		audio_filenames += temp_wavs
		roi_filenames += temp_rois
	# Reproducibly shuffle.
	audio_filenames = np.array(audio_filenames)
	roi_filenames = np.array(roi_filenames)
	perm = np.argsort(audio_filenames)
	audio_filenames, roi_filenames = audio_filenames[perm], roi_filenames[perm]
	if shuffle:
		np.random.seed(42)
		perm = np.random.permutation(len(audio_filenames))
		audio_filenames = audio_filenames[perm]
		roi_filenames = roi_filenames[perm]
		np.random.seed(None)
	# Split.
	i = int(round(split * len(audio_filenames)))
	return { \
		'train': { \
			'audio': audio_filenames[:i], 'rois': roi_filenames[:i]}, \
		'test': { \
			'audio': audio_filenames[i:], 'rois': roi_filenames[i:]} \
		}

def get_simsiam_loaders_motif(partition, p, batch_size=64, \
	shuffle=(True, False), num_workers=4, n_samples=2048,min_spec_val=None):
	"""
	Get DataLoaders for training and testing: fixed-duration shotgun VAE

	Parameters
	----------
	partition : dict
		Output of ``ava.models.window_vae_dataset.get_window_partition``.
	p : dict
		Preprocessing parameters. Must contain keys: ...
	batch_size : int, optional
		Defaults to ``64``.
	shuffle : tuple of bool, optional
		Whether to shuffle train and test sets, respectively. Defaults to
		``(True, False)``.
	num_workers : int, optional
		Number of CPU workers to feed data to the network. Defaults to ``4``.

	Returns
	-------
	loaders : dict
		Maps the keys ``'train'`` and ``'test'`` to their respective
		DataLoaders.
	"""
	train_dataset = simsiamSet(partition['train']['audio'], \
			partition['train']['rois'], p, transform=numpy_to_tensor, \
			min_spec_val=min_spec_val,dataset_length=n_samples)
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, \
			shuffle=shuffle[0], num_workers=num_workers)
	if not partition['test']:
		return {'train':train_dataloader, 'test':None}
	test_dataset = simsiamSet(partition['test']['audio'], \
			partition['test']['rois'], p, transform=numpy_to_tensor, \
			min_spec_val=min_spec_val,dataset_length=n_samples)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, \
			shuffle=shuffle[1], num_workers=num_workers)
	return {'train':train_dataloader, 'test':test_dataloader}

class simsiamSet(Dataset):

	def __init__(self, audio_filenames, roi_filenames, p, transform=None,
		dataset_length=2048, min_spec_val=None,max_len = 300,adult=True):
		"""
		Create a torch.utils.data.Dataset for chunks of animal vocalization.

		Parameters
		----------
		audio_filenames : list of str
			List of wav files.
		roi_filenames : list of str
			List of files containing animal vocalization times.
		transform : {``None``, function}, optional
			Transformation to apply to each item. Defaults to ``None`` (no
			transformation).
		dataset_length : int, optional
			Arbitrary number that determines batch size. Defaults to ``2048``.
		min_spec_val : {float, None}, optional
			Used to disregard silence. If not `None`, spectrogram with a maximum
			value less than `min_spec_val` will be disregarded.
		"""
		self.filenames = np.array(sorted(audio_filenames))
		#print(self.filenames[0])
		self.adult = adult
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=WavFileWarning)
			self.audio = [wavfile.read(fn)[1] for fn in self.filenames]
			self.fs = wavfile.read(audio_filenames[0])[0]
		self.roi_filenames = roi_filenames
		self.min_spec_val = min_spec_val
		self.p = p
		self.train_augment=p['train_augment']
		self.max_tau = p['max_tau']
		self.rois = [np.loadtxt(i, ndmin=2) for i in roi_filenames]
		self.file_weights = np.array([np.sum(np.diff(i)) for i in self.rois])
		self.file_weights /= np.sum(self.file_weights)
		self.roi_weights = []
		self.max_len = max_len
		curr_sum = -1
		self.fsum = np.zeros(len(self.roi_filenames)).astype(np.int32)
		self.dt = self.p['window_length'] - self.p['window_overlap']
		for i in range(len(self.rois)):
			temp = np.diff(self.rois[i]).flatten()
			self.roi_weights.append(temp/np.sum(temp))
			curr_sum += len(self.rois[i])
			#print(curr_sum)
			self.fsum[i] = int(abs(curr_sum))

		self.transform = transform
		#print(self.fsum)
		self.dataset_length_real = int(self.fsum[-1])
		self.dataset_length_fake = dataset_length
		#print(self.dataset_length)

	def __len__(self):
		"""NOTE: length is arbitrary"""
		if self.train_augment:
			return self.dataset_length_fake
		else:
			return self.dataset_length_real


	def __getitem__(self, index, seed=None, shoulder=0.05, \
		return_seg_info=False):
		"""
		Get spectrograms.

		Parameters
		----------
		index :
		seed :
		shoulder :
		return_seg_info :

		Returns
		-------
		specs :
		file_indices :
		onsets :
		offsets :
		"""
		specs1, file_indices, onsets, offsets,specs2 = [], [], [], [],[]
		single_index = False
		try:
			iterator = iter(index)
		except TypeError:
			index = [index]
			single_index = True
		np.random.seed(seed)
		if self.train_augment:
			for i in index:
				while True:
					# First find the file, then the ROI.
					file_index = np.random.choice(np.arange(len(self.filenames)), \
						p=self.file_weights)
					load_filename = self.filenames[file_index]
					roi_index = \
						np.random.choice(np.arange(len(self.roi_weights[file_index])),
						p=self.roi_weights[file_index])
					roi = self.rois[file_index][roi_index]
					# Then choose a chunk of audio uniformly at random.
					onset = roi[0] + (roi[1] - roi[0] - self.p['window_length']) \
						* np.random.rand()
					onset2 = onset + random.uniform(-self.max_tau,self.max_tau)
					offset = onset + self.p['window_length']
					offset2 = onset2 + self.p['window_length']
					target_times = np.linspace(onset, offset, \
						self.p['num_time_bins'])
					target_times2 = np.linspace(onset2, offset2, \
						self.p['num_time_bins'])
					# Then make a spectrogram.
					spec, flag = self.p['get_spec'](max(0.0, onset-shoulder), \
						offset+shoulder, self.audio[file_index], self.p, \
						fs=self.fs, target_times=target_times)
					spec2, flag2 = self.p['get_spec'](max(0.0, onset2-shoulder), \
						offset2+shoulder, self.audio[file_index], self.p, \
						fs=self.fs, target_times=target_times2)
					if not flag:
						continue
					if not flag2:
						continue
				# Remake the spectrogram if it's silent.
					if self.min_spec_val is not None and \
						np.max(spec) < self.min_spec_val:
						continue
					if self.min_spec_val is not None and \
						np.max(spec2) < self.min_spec_val:
						continue
					if self.transform:
						spec = self.transform(spec)
						spec2 = self.transform(spec2)
					specs1.append(spec)
					file_indices.append(file_index)
					onsets.append(onset)
					offsets.append(offset)
					specs2.append(spec2)
					break
		else:
			
			for i in index:
				st = []
				# First find the file, then the ROI.
				roi_index = i#np.random.choice(np.arange(len(self.filenames)), \
				#p=self.file_weights)


				file_index = np.where(i <= self.fsum)[0][0]
				#print('file index: ',file_index)
				#print('selected cumsum: ', self.fsum[file_index])
				#print('next cumsum: ',self.fsum[file_index + 1])
				#print('index: ', i)
				load_filename = self.filenames[file_index]
				if self.adult:
					day = 1
				else:
					day = int(load_filename.split('/')[-3]) # or something like this. deal with this later
				rois = self.rois[file_index]
				#print('index:', i)
				#print('cumsum:',self.fsum[file_index])
				in_file_ind = len(rois) - 1 - (self.fsum[file_index] - i)
				#print('Index in file:',in_file_ind)
				#print(len(rois))
				#print('Actual index:',i)
				#print('cumsum of files:',self.fsum[file_index])
				onset = rois[in_file_ind][0]
				offset = rois[in_file_ind][1]
				if offset-self.p['window_length']-onset < 0:
					ons = [onset]
				else:
					ons = np.linspace(onset,offset-self.p['window_length'],\
						num = int((offset-onset)//(self.p['window_length'] - self.p['window_overlap'])))
					num = int((offset-onset)//(self.p['window_length'] - self.p['window_overlap']))
					
					ons = ons[:len(ons)]
						# Then choose a chunk of audio uniformly at random.
				for ton in ons:

					toff = ton + self.p['window_length']
					#offset = roi[1]
					target_times = np.linspace(ton, toff, \
						self.p['num_time_bins'])
					# Then make a spectrogram.
					spec, flag = self.p['get_spec'](max(0.0, ton-shoulder), \
						toff+shoulder, self.audio[file_index], self.p, \
						fs=self.fs, target_times=target_times)
					if self.transform:
						spec = self.transform(spec)

					st.append(spec)

				specs1.append(st)
				specs2.append(1)
				file_indices.append(file_index)
				onsets.append(onset)
				offsets.append(offset)
				#days.append(day)

		np.random.seed(None)
		if return_seg_info:
			if single_index:
					return specs1[0], file_indices[0], onsets[0], offsets[0]
			return specs1, file_indices, onsets, offsets
		if single_index:
			return (specs1[0],specs2[0])
		return (specs1,specs2)

############## Smoothness analysis functions ###########

def total_change(list_of_changes,type='l1'):

    '''
    takes as input a list of changes within trajectory, outputs 
    a list of metrics indicating total amount of change within trajectory
    and mean changes within trajectory
    '''

    tc = []
    tcm = []

    if type == 'l1':
        for traj in list_of_changes:
            changes = np.sum(np.abs(traj),axis=1)
            tc.append(changes)
            tcm.append(np.mean(changes))

    elif type == 'l2':
        for traj in list_of_changes:
            changes = np.sum(traj ** 2,axis=1)
            tc.append(changes)
            tcm.append(np.mean(changes))

    elif type == 'linf':
        for traj in list_of_changes:
            changes = np.argmax(traj,axis=1)
            tc.append(changes)
            tcm.append(np.mean(changes))
    else: 
        print('Not implemented!')
        raise NotImplementedError


    return tc,tcm


def cosine_angle_change(list_of_trajs):

    '''
    takes as input a list of latent vectors within a trajectory, outputs a list of cosine similarities,angles between vectors
    in addition to averages of the two
    '''

    cos_sims = []
    angles = []
    mean_cos_sims = []
    mean_angles = []

    for traj in list_of_trajs:
        tmp_cos = []
        tmp_ang = []

        for vec_ind in range(traj.shape[0] - 1):
            v1 = traj[vec_ind,:]
            v2 = traj[vec_ind +1,:]

            num = v1 @ v2 
            denom = np.sqrt(v1 @ v1) * np.sqrt(v2 @ v2)

            cos = num/denom
            tmp_cos.append(cos)

            ang = math.degrees(math.acos(cos))
            tmp_ang.append(ang)

        cos_sims.append(tmp_cos)
        mean_cos_sims.append(np.mean(tmp_cos))

        angles.append(tmp_ang)
        mean_angles.append(np.mean(tmp_ang))

    return (angles, cos_sims), (mean_angles,mean_cos_sims)




#####################  MMD Functions ################
def rbf_dot(patterns1,patterns2,sig):

    size1 = patterns1.shape
    size2 = patterns2.shape

    G = np.pow(patterns1,2).sum(axis=1)
    H = np.pow(patterns2,2).sum(axis=1)

    Q = np.tile(G,[1,size2[0]])
    R = np.tile(H.T,[size1[0],1])

    H = Q + R - 2*patterns1 @ patterns2.T

    H=np.exp(-H/2/sig**2);

    return H 

def mmd_fxn(lat1,lat2,sig):

    if sig == -1:
        Z = np.hstack([lat1,lat2])

        size1 = Z.shape[0]
        if size1 > 100:
            Zmed = Z[0:100,:]
            size1 = 100
        else:
            Zmed = Z 

        G = np.sum(Zmed **2,axis=1)
        Q = np.tile(G,[1,size1])
        R = np.tile(G.T,[size1,1])

        dists = Q + R - 2* Zmed @ Zmed.T
        dists = dists - np.tril(dists)
        dists = np.reshape(dists,[size1**2,1])
        sig = np.sqrt(0.5 * np.median(dists[dists > 0]))

    mmds = []
    for l in lat1:
        l = np.expand_dims(l,0)
        mmd = np.sum(rbf_dot(l,lat1) - rbf_dot(l,lat2))
        mmds.append(mmd)

    return mmds

############# Dataframe functions ###############
def numpy_to_tensor(x):
	"""Transform a numpy array into a torch.FloatTensor."""
	return torch.from_numpy(x).type(torch.FloatTensor)


############### Old dataframe & latent dynamics fxns - necessary?? #########

def stitch_images(list_of_images,thresh=0.05):

	# initialize with first image
	i1 = list_of_images[0]

	# iterate through images to stitch
	orig_total_len = i1.shape[1]
	for ind in range(len(list_of_images) - 1):
		i2 = list_of_images[ind + 1]

		orig_total_len += i2.shape[1]

		ldif = i1.shape[1] - i2.shape[1]

		corr_vec = np.zeros((i2.shape[1],))
		for offset in range(i2.shape[1]):

			tmpvec_i1 = i1[:,ldif + offset::].flatten()
			if offset == 0:
				tmpvec_i2 = i2.flatten()
			else:
				tmpvec_i2 = i2[:,0:-offset].flatten()
			# calculate correlation of overlap
			a = (tmpvec_i1 - np.mean(tmpvec_i1))/(np.std(tmpvec_i1)* len(tmpvec_i1))
			b = (tmpvec_i2 - np.mean(tmpvec_i2))/np.std(tmpvec_i2)
			corr_vec[offset] = np.correlate(a,b)

		max_corr_ind = np.argmax(corr_vec)
		if corr_vec[max_corr_ind] > thresh:
			#print('Overlap detected! Stitching from offset {0}'.format(max_corr_ind))
			#best_offset = max_corr_ind
			if max_corr_ind == 0:
			#	print('stitchin')
			#	print('old shape: {0}'.format(i1.shape))

				i1[:,ldif::] = (i1[:,ldif::] + i2)/2
			#	print('new shape: {0}'.format(i1.shape))
			else:
				i1[:,ldif+max_corr_ind::] = (i1[:,ldif+max_corr_ind::] + i2[:,0:-max_corr_ind])/2

				i1 = np.hstack((i1, i2[:,max_corr_ind::]))

		else:
			i1 = np.hstack((i1,i2))
			#print('No overlap! Concatenating')
	new_total_len = i1.shape[1]

	print('original total length: {0} \n new total length: {1}'.format(orig_total_len,new_total_len))
	return i1


#@numba.njit(fastmath=True)
#@numba.njit(fastmath = False)
def stitch_and_decode(model,trajectories, day,path='.'):

	#for trajectory in trajectories:

	#audio = []
#	stitcher = cv2.Stitcher.create(mode = 1)
	max_decode = 50
	if not os.path.isdir(path):
		os.mkdir(path)

	specs = []
	spec_ind = 0
	print('decoding trajectories')
	print(trajectories.shape)
	for trajectory in trajectories:

		trajectory = torch.tensor(trajectory).type(torch.FloatTensor).to(model.device)
		if trajectory.shape[1] != 32:
			trajectory = trajectory.T[:,:32]
		#print(trajectory.shape)
		tmpspecs = []
		with torch.no_grad():
			#print(trajectory.shape)
			traj_rec = model.decode(trajectory).view(-1, X_SHAPE[0], X_SHAPE[1]).detach().cpu().numpy()
			#print(np.range(traj_rec))
			#print(np.min(traj_rec))
			#####print(np.max(traj_rec))
			#traj_rec -= 2
			#traj_rec /= 5
			#traj_rec = np.clip(traj_rec, 0.0, 1.0)
			for t in traj_rec:
				#gray = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
				#gray_three = cv2.merge([t,t,t])
				tmpspecs.append(t)

			result = stitch_images(tmpspecs)
			#result = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
			#if status != cv2.Stitcher_OK:
			#	print('uh oh')
			specs.append(result)
			fig, ax = plt.subplots()
			ax.imshow(np.flipud(result))
			ax.set_aspect('auto')
			plt.savefig(os.path.join(path,'simulated_spec_'+ str(day) + 'dph_' + str(spec_ind) + '.png'))
			plt.close(fig)
			spec_ind += 1

		if spec_ind >= max_decode:
			break

	#print(np.hstack(tmpspecs).shape)
	'''
	print('writing audio')
	spec_ind = 0
	for spec in specs:
		tmp_aud = librosa.core.spectrum.griffinlim(spec)
		scipy.io.wavfile.write(os.path.join(path,'simulated_audio_' + str(spec_ind) + '.wav'),\
					FS,np.array(tmp_aud,np.int16))
		spec_ind += 1
		#audio.append(tmp_aud)
		### reshape to be many T by 128
		### turn to .wav
		### return spectrogram
	'''
	return specs

def latents_to_pickle(latents, list_of_latent_days,path,n_per_file=30,just_read=False,max_files=300):

	if not os.path.isdir(path):
		os.mkdir(path)

	if just_read:

		fnames = glob.glob(os.path.join(path,'*.pkl'))
		voc_data = {
			'latent_fnames':fnames,
			'n_files':len(fnames),
			}
	else:
		write_file_num = 0
		voc_data = {
			'latent_fnames':[],
			'n_files':0,
			}
		n_in_file = 1
		save_filename = \
			"latent_vocs_" + str(write_file_num).zfill(4) + '.pkl'
		save_filename = os.path.join(path, save_filename)

		data_to_pickle = {}

		latent_in_file = []
		length_voc = []
		latent_day = []
		voc_data['latent_fnames'].append(save_filename)
		voc_data['n_files'] += 1
		for ind,latent in enumerate(latents):


			if n_in_file <= n_per_file:
				latent_in_file.append(latent)
				length_voc.append(len(latent))
				latent_day.append(list_of_latent_days[ind])
				n_in_file += 1
			else:

				write_file_num += 1
				#print(len(latent_in_file))
				#print(latent_in_file)
				data_to_pickle['latent_in_file'] = latent_in_file
				data_to_pickle['length_voc'] = length_voc
				data_to_pickle['latent_day'] = latent_day
				data_to_pickle['length_file'] = n_per_file
				with open(save_filename, "wb") as f:
					pickle.dump(data_to_pickle,f)

				latent_in_file = []
				length_voc = []
				data_to_pickle = {}
				voc_data['latent_fnames'].append(save_filename)
				voc_data['n_files'] += 1
				save_filename = \
					"latent_vocs_" + str(write_file_num).zfill(4) + '.pkl'
				save_filename = os.path.join(path, save_filename)

				n_in_file = 1

			if write_file_num >= max_files:
				return voc_data

		if len(latent_in_file) > 0:
			data_to_pickle['latent_in_file'] = latent_in_file
			data_to_pickle['length_voc'] = length_voc
			data_to_pickle['latent_day'] = latent_day
			data_to_pickle['length_file'] = len(latent_in_file)
			with open(save_filename, "wb") as f:
				pickle.dump(data_to_pickle,f)

			voc_data['latent_fnames'].append(save_filename)
			voc_data['n_files'] += 1

	return voc_data

############ Plotting Functions ###########

def set_colors(vec,min,max):

	mapped_vals = []
	#min mako: (0.1819,0.119,0.231) #2e1e3b
	#max mako: (0.546,0.854,0.698) #8bdab3

	cmin = Color('#2e1e3b')
	cmax = Color('#8bdab3')

	val_range = np.linspace(-1,1,500)
	crange = list(cmin.range_to(cmax,500))
	for data in vec:
		val = 2 * (data - min)/max - 1 # range between zero and 1
		min_ind = np.argmin(np.abs(val_range - val))

		mapped_vals.append(val_range[min_ind])


	return mapped_vals





##############################################
if __name__ == '__main__':

    pass