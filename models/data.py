import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from scipy.interpolate import interp2d
import warnings
from scipy.io.wavfile import WavFileWarning

EPSILON = 1e-12

def z_score(data):

	x_stacked =np.vstack(data)
	mu = np.nanmean(x_stacked,axis=0,keepdims=True)
	sd = np.nanstd(x_stacked,axis=0,keepdims=True)

	return [(d - mu)/sd for d in data],mu,sd

def scale(data):
	x_stacked =np.vstack(data)
	sd = np.nanstd(x_stacked,axis=0,keepdims=True)
	mag = np.amax(np.abs(x_stacked))
	return [d/mag for d in data],mag

def generate_ndim_benes(n=100,d = 20,T=100,dt=1):

	t = np.arange(0,T,dt)
	
	allPaths = []

	for ii in range(n):

		sample_dW = dt * np.random.randn(len(t),d)
		xnot = np.zeros((d,))#0.01*np.random.randn(d)

		xx = [xnot]
		for jj in range(len(t)):

			dx = np.tanh(xx[jj])*dt + sample_dW[jj]

			xx.append(xx[jj] + dx)

		xx = np.vstack(xx)
		assert xx.shape[0] == (len(t) + 1), print(xx.shape)
		allPaths.append(xx)

	return allPaths

def generate_geometric_brownian(n=100,T=100,dt=1,mu=1,sigma=0.5,x0=0.1):

	allPaths=[]
	t = np.arange(0,T,dt)
	#print("not adding noise")
	def f(x,t):
		return mu*x
	def g(x,t):
		return sigma*x
	
	def dW(dt):
		return np.random.normal(loc=0.,scale=np.sqrt(dt))
	for ii in range(n):

		xnot = x0 + 0.03**2 * np.random.randn(1)
		xx = [xnot]

		for jj in range(1,len(t)+1):
			x = xx[jj-1]
			tt = t[jj-1]
			xx.append(x + f(x,tt)*dt + g(x,tt)*dW(dt))
		xx = np.hstack(xx)[:,None]

		#xx = xx + (0.01)*(dt/0.02)*np.random.randn(*xx.shape)
		
		allPaths.append(xx)

	mu_fnc = lambda x: mu * x.detach().cpu().numpy() 
	d_fnc = lambda x: sigma * x.detach().cpu().numpy()

	eta_fnc = lambda x: mu_fnc(x)/d_fnc(x)**2
	lam_fnc = lambda x: 1/d_fnc(x)

	return allPaths,(mu_fnc,d_fnc),(eta_fnc,lam_fnc)

def generate_vanderpol(n=100,T = 1, dt=0.001,rho=2,tau=15,sigma=0.25,x0=np.array([1,1])):

	allPaths=[]
	t = np.arange(0,T,dt)
	#print("not adding noise")
	sig = np.eye(2) * sigma
	lam = np.eye(2)/sigma
	def f(x,t):
		dx1 = rho * tau * (x[0] - x[0]**3/3 - x[1])
		dx2 = tau/rho * x[0]
		return np.hstack([dx1,dx2])
	def g(x,t):
		return sigma*x
	
	def dW(dt):
		return np.random.normal(loc=np.zeros((2,)),scale=np.sqrt(dt))
	
	for ii in range(n):

		xnot = x0 + 0.03**2 * np.random.randn(2)
		xx = [xnot]

		for jj in range(1,len(t)+1):

			x = xx[jj-1]
			tt = t[jj-1]
			xx.append(x + f(x,tt)*dt + g(x,tt)*dW(dt))

		xx = np.vstack(xx)
		assert xx.shape[0] == (len(t) + 1), print(xx.shape)
		
		allPaths.append(xx)

	inds = np.tril_indices(3)
	mu_fnc = lambda x,dt=0.001: np.vstack([rho * tau * (x[:,0] - x[:,0]**3/3 - x[:,1]).detach().cpu().numpy() * dt,
										(rho/tau * x[:,0]*dt).detach().cpu().numpy()*dt ]).T
	d_fnc = lambda x: np.sqrt(dt)*torch.FloatTensor(sig)[None,:,:].repeat(x.shape[0],1,1).detach().cpu().numpy()

	eta_fnc = lambda x:  np.vstack([rho * tau * (x[:,0] - x[:,0]**3/3 - x[:,1]).detach().cpu().numpy() * dt,
										(rho/tau * x[:,0]*dt).detach().cpu().numpy()*dt ]).T
	lam_fnc = lambda x: torch.FloatTensor(lam)[None,:,:].repeat(x.shape[0],1,1).detach().cpu().numpy()/np.sqrt(dt)

	return allPaths,(mu_fnc,d_fnc),(eta_fnc,lam_fnc)

def generate_2d_swirls(n=100,T=1,dt=0.001,mu=25,sigma=0.5,x0=np.array([0.1,0.1])):

	allPaths=[]
	t = np.arange(0,T,dt)
	#print("not adding noise")
	
	def f(x,t):
		dx1 = -mu * x[1]
		dx2 = mu * x[0]
		#assert np.all(dx == dx2)
		return np.hstack([dx1,dx2])
	def g(x,t):
		return sigma*x
	
	def dW(dt):
		return np.random.normal(loc=np.zeros((2,)),scale=np.sqrt(dt))
	for ii in range(n):

		xnot = x0 + 0.03**2 * np.random.randn(2)
		xx = [xnot]

		

		for jj in range(1,len(t)+1):
			x = xx[jj-1]
			tt = t[jj-1]
			xx.append(x + f(x,tt)*dt + g(x,tt)*dW(dt))
		xx = np.vstack(xx)
		#xx = xx + (0.01)*(dt/0.02)*np.random.randn(*xx.shape)
		
		allPaths.append(xx)

	sigMat = np.eye(2) * sigma
	LMat = np.eye(2) / sigma
	inds = np.tril_indices(2)
	mu_fnc = lambda x: (f(x,0)).squeeze()
	d_fnc = lambda x: torch.diag_embed(sigma * x).detach().cpu().numpy()[:,inds[0],inds[1]]

	eta_fnc = lambda x: ((LMat @ x[:,:,None].detach().cpu().numpy())**2 * (A @ x[:,:,None].detach().cpu().numpy())).squeeze()
	lam_fnc = lambda x: (torch.diag_embed(1/(sigma * x))).detach().cpu().numpy()[:,inds[0],inds[1]]
	return allPaths,(mu_fnc,d_fnc),(eta_fnc,lam_fnc)

def generate_stochastic_lorenz(n=100,T=1,dt=0.001,coeffs=[10,28,8/3,0.15,0.15,0.15],zscore=False):

	sigma,rho,beta = coeffs[0],coeffs[1],coeffs[2]
	A = np.array([coeffs[3],coeffs[4],coeffs[5]])
	Sig = np.eye(3) * A
	t = np.arange(0,T,dt)
	
	allPaths = [ ]

	def f(x,t):
		dx = sigma * (x[1] - x[0]) #+ sample_dW[0]
		dy = (x[0] * (rho - x[2]) - x[1]) #+ sample_dW[1]
		dz = (x[0]*x[1]  - beta*x[2]) #+ sample_dW[2]
		return np.hstack([dx,dy,dz])
	
	def g(x,t):
		return A[0]
	
	def dW(dt):
		return np.random.normal(loc=np.zeros((3,)),scale=np.sqrt(dt))
	
	for ii in range(n):

		
		xnot = np.random.randn(3)

		xx = [xnot]
		for jj in range(1,len(t)+1):

			x = xx[jj-1]
			tt = t[jj-1]
			xx.append(x + f(x,tt)*dt + g(x,tt)*dW(dt))

		xx = np.vstack(xx)
		assert xx.shape[0] == (len(t) + 1), print(xx.shape)
		
		allPaths.append(xx)

	if zscore:
		stacked = np.vstack(allPaths)
		mu = np.nanmean(stacked,axis=0)
		sd = np.nanstd(stacked,axis=0)
		allPaths = [(x - mu)/sd for x in allPaths]
	inds = np.tril_indices(3)
	
	mu_fnc = lambda x,dt=0.001: np.vstack([(sigma * (x[:,1] - x[:,0])).detach().cpu().numpy() * dt,
										(x[:,0]*(rho - x[:,2]) - x[:,1]).detach().cpu().numpy() * dt,
										(x[:,0]*x[:,1] - beta*x[:,2]).detach().cpu().numpy() * dt]).T
	d_fnc = lambda x: np.sqrt(dt)*torch.FloatTensor(A)[None,:,:].repeat(x.shape[0],1,1).detach().cpu().numpy()[:,inds[0],inds[1]]

	eta_fnc = lambda x: np.vstack([(sigma * (x[:,1] - x[:,0])).detach().cpu().numpy() /A1**2,
										(x[:,0]*(rho - x[:,2]) - x[:,1]).detach().cpu().numpy() /A2**2,
										(x[:,0]*x[:,1] - beta*x[:,2]).detach().cpu().numpy() /A3**2]).T
	lam_fnc = lambda x: torch.FloatTensor(1/A)[None,:,:].repeat(x.shape[0],1,1).detach().cpu().numpy()[:,inds[0],inds[1]]/np.sqrt(dt)
	return allPaths,(mu_fnc,d_fnc),(eta_fnc,lam_fnc)


def downsample(data:list,origdt:float,newdt:float,noise:bool=True) -> np.ndarray:

	skip = int(newdt/origdt)

	downsampled = [d[::skip] for d in data]

	if noise:
		downsampled = [d + 0.01*np.random.randn(*d.shape) for d in downsampled]

	return downsampled

class toyDataset(Dataset):

	def __init__(self,data,dt) -> None:
		"""
		toyData: list of numpy arrays
		"""

		lens = list(map(len,data))
		lens = [0] + list(np.cumsum([l for l in lens][:-1]))
		#lenall.append([l - 1 for l in lens][:-1])
		validInds = np.hstack([list(range(l,len(t)-1 + l)) \
			 for l,t in zip(lens,data)])
	
		self.data= np.vstack(data)
		self.data_inds = validInds
		self.dt = dt
		self.length = len(validInds)
		## needed: slice data by dt? need true dt, ds dt for that
		## should be fine to add though

	def __len__(self):

		return self.length 
	
	def __getitem__(self, index):
		
		single_index = False
		result = []
		try:
			iterator = iter(index)
		except TypeError:
			index = [index]
			single_index = True

		for ii in index:
			ind1 = self.data_inds[ii]
			ind2 = ind1 + 1
			s1,s2 = self.transform(self.data[ind1]),self.transform(self.data[ind2])
			result.append((s1,s2,self.dt))

		if single_index:
			return result[0]
		return result
	
	def transform(self,data):
		return torch.from_numpy(data).type(torch.FloatTensor)


		
class FixedWindowDataset(Dataset):

	def __init__(self, audio_filenames, roi_filenames, p,
		dataset_length=2048, min_spec_val=None,dt=0.05,win_length=0.05,overlap=0.5):
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
		dt : float, optional
			timestep between successive 
		"""
		self.filenames = np.array(sorted(audio_filenames))
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=WavFileWarning)
			self.audio = [wavfile.read(fn)[1] for fn in self.filenames]
			self.fs = wavfile.read(audio_filenames[0])[0]
		self.roi_filenames = roi_filenames
		self.dataset_length = dataset_length
		self.min_spec_val = min_spec_val
		self.win_length=win_length
		self.rois = [np.loadtxt(i, ndmin=2) for i in roi_filenames]
		self.file_weights = np.array([np.sum(np.diff(i)) for i in self.rois])
		self.file_weights /= np.sum(self.file_weights)
		self.roi_weights = []
		self.dt = win_length*(1 - overlap) 
		self.win_length = win_length
		self.overlap = overlap
		self.p = p
		for i in range(len(self.rois)):
			temp = np.diff(self.rois[i]).flatten()
			self.roi_weights.append(temp/np.sum(temp))


	def __len__(self):
		"""NOTE: length is arbitrary"""
		return self.dataset_length


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
		specs, specs2,dts,file_indices, onsets, offsets = [],[], [],[], [], []
		single_index = False
		try:
			iterator = iter(index)
		except TypeError:
			index = [index]
			single_index = True
		np.random.seed(seed)
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
				onset = roi[0] + (roi[1] - roi[0] - 2*self.win_length) \
					* np.random.rand()
				offset = onset + self.win_length

				onset2 = onset + self.dt 
				offset2 = onset2 + self.win_length
				target_times = np.linspace(onset, offset, \
						self.p['num_time_bins'])
				target_times2 = np.linspace(onset2, offset2, \
						self.p['num_time_bins'])
				# Then make a spectrogram.
				spec, flag = get_spec(max(0.0, onset-shoulder), \
						offset+shoulder, self.audio[file_index], self.p, \
						fs=self.fs, target_times=target_times)
				if not flag:
					continue
				# Remake the spectrogram if it's silent.
				if self.min_spec_val is not None and \
						np.max(spec) < self.min_spec_val:
					continue

				spec2, flag2 = get_spec(max(0.0, onset2-shoulder), \
						offset2+shoulder, self.audio[file_index], self.p, \
						fs=self.fs, target_times=target_times2)
				
				spec = self.transform(spec).view(1,spec.shape[0],spec.shape[1])
				spec2 = self.transform(spec2).view(1,spec2.shape[0],spec2.shape[1])
				specs.append(spec)
				specs2.append(spec2)
				file_indices.append(file_index)
				onsets.append(onset)
				offsets.append(offset)
				dts.append(self.dt)
				break
		np.random.seed(None)
		if return_seg_info:
			if single_index:
				return specs[0], specs2[0],dts[0], file_indices[0], onsets[0], offsets[0]
			return specs, specs2,dts,file_indices, onsets, offsets
		if single_index:
			return specs[0],specs2[0],dts[0]
		return specs,specs2,dts

	def transform(self,data):
		return torch.from_numpy(data).type(torch.FloatTensor)
	
def makeToyDataloaders(ds1,ds2,dt,batch_size=512):

	#assert ds1.shape[1] == 3
	#ds1 = ds1).type(torch.FloatTensor)
	#ds2 = torch.from_numpy(ds2).type(torch.FloatTensor)
	dataset1 = toyDataset(ds1,dt)
	dataset2 = toyDataset(ds2,dt)

	trainDataLoader = DataLoader(dataset1,batch_size=batch_size,shuffle=True,
			      num_workers=4)
	testDataLoader = DataLoader(dataset2,batch_size=batch_size,shuffle=False,
			      num_workers=4)
	
	return {'train':trainDataLoader,'test':testDataLoader}


def compareSwirlsModels(nTrajs, dim,muFnc, sigFnc, model,dt=0.001,T=1):

	trueTrajs = []
	modelTrajs=[]
	t = np.arange(0,T,dt)
	for traj in range(nTrajs):
		xnot = 0.1 + 0.03**2 * np.random.randn(dim)
		noisePath = np.random.randn((dim,len(t)))*np.sqrt(dt)
		
		xxTrue = [xnot]
		xxModel = [torch.from_numpy(xnot).type(torch.FloatTensor).to(model.device)]
		for time in range(len(t)):


			xPrevTrue = xxTrue[time]
			xPrevModel = xxModel[time]

			dzTrue = muFnc(xPrevTrue)*dt + noisePath[:,time] * sigFnc(xPrevTrue)
			mu,L = model.getMoments(xPrevModel)
			dzModel = mu * dt + L @ noisePath[:,time]

			xxTrue.append(xxTrue[time] + dzTrue)
			xxModel.append(xxModel[time] + dzModel)

		xxModel = [x.detach().cpu().numpy() for x in xxModel]

		trueTrajs.append(np.array(xxTrue))
		modelTrajs.append(np.array(xxModel))

	return trueTrajs,modelTrajs

def get_spec(t1, t2, audio, p, fs=32000, target_freqs=None, target_times=None, \
	fill_value=-1/EPSILON, max_dur=None, remove_dc_offset=True):
	"""
	Norm, scale, threshold, stretch, and resize a Short Time Fourier Transform.

	Notes
	-----
	* ``fill_value`` necessary?
	* Look at all references and see what can be simplified.
	* Why is a flag returned?

	Parameters
	----------
	t1 : float
		Onset time.
	t2 : float
		Offset time.
	audio : numpy.ndarray
		Raw audio.
	p : dict
		Parameters. Must include keys: ...
	fs : float
		Samplerate.
	target_freqs : numpy.ndarray or ``None``, optional
		Interpolated frequencies.
	target_times : numpy.ndarray or ``None``, optional
		Intepolated times.
	fill_value : float, optional
		Defaults to ``-1/EPSILON``.
	max_dur : float, optional
		Maximum duration. Defaults to ``None``.
	remove_dc_offset : bool, optional
		Whether to remove any DC offset from the audio. Defaults to ``True``.

	Returns
	-------
	spec : numpy.ndarray
		Spectrogram.
	flag : bool
		``True``
	"""
	if max_dur is None:
		max_dur = p['max_dur']
	if t2 - t1 > max_dur + 1e-4:
		message = "Found segment longer than max_dur: " + str(t2-t1) + \
				"s, max_dur = " + str(max_dur) + "s"
		#warnings.warn(message)
	s1, s2 = int(round(t1*fs)), int(round(t2*fs))
	assert s1 < s2, "s1: " + str(s1) + " s2: " + str(s2) + " t1: " + str(t1) + \
			" t2: " + str(t2)
	# Get a spectrogram and define the interpolation object.
	temp = min(len(audio),s2) - max(0,s1)
	if temp < p['nperseg'] or s2 <= 0 or s1 >= len(audio):
		return np.zeros((p['num_freq_bins'], p['num_time_bins'])), True
	else:
		temp_audio = audio[max(0,s1):min(len(audio),s2)]
		if remove_dc_offset:
			temp_audio = temp_audio - np.mean(temp_audio)
		f, t, spec = stft(temp_audio, fs=fs, nperseg=p['nperseg'], \
				noverlap=p['noverlap'])
	t += max(0,t1)
	spec = np.log(np.abs(spec) + EPSILON)
	interp = interp2d(t, f, spec, copy=False, bounds_error=False, \
		fill_value=fill_value)
	# Define target frequencies.
	if target_freqs is None:
		if p['mel']:
			target_freqs = np.linspace(_mel(p['min_freq']), \
					_mel(p['max_freq']), p['num_freq_bins'])
			target_freqs = _inv_mel(target_freqs)
		else:
			target_freqs = np.linspace(p['min_freq'], p['max_freq'], \
					p['num_freq_bins'])
	# Define target times.
	if target_times is None:
		duration = t2 - t1
		if p['time_stretch']:
			duration = np.sqrt(duration * max_dur) # stretched duration
		shoulder = 0.5 * (max_dur - duration)
		target_times = np.linspace(t1-shoulder, t2+shoulder, p['num_time_bins'])
	# Then interpolate.
	interp_spec = interp(target_times, target_freqs, assume_sorted=True)
	spec = interp_spec
	# Normalize.
	spec -= p['spec_min_val']
	spec /= (p['spec_max_val'] - p['spec_min_val'])
	spec = np.clip(spec, 0.0, 1.0)
	# Within-syllable normalize.
	if p['within_syll_normalize']:
		spec -= np.quantile(spec, p['normalize_quantile'])
		spec[spec<0.0] = 0.0
		spec /= np.max(spec) + EPSILON
	return spec, True



def _mel(a):
	"""https://en.wikipedia.org/wiki/Mel-frequency_cepstrum"""
	return 1127 * np.log(1 + a / 700)


def _inv_mel(a):
	"""https://en.wikipedia.org/wiki/Mel-frequency_cepstrum"""
	return 700 * (np.exp(a / 1127) - 1)




