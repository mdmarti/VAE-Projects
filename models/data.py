import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
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
	for ii in range(n):

		xnot = x0 + 0.03**2 * np.random.randn(1)
		xx = [xnot]

		for jj in range(1,len(t)+1):
			x = xx[jj-1]
			sample_dW = np.sqrt(dt) * np.random.randn(1)
			dx = mu*x *dt + sigma*x * sample_dW
			xx.append(xx[jj-1] + dx)
		xx = np.hstack(xx)[:,None]

		xx = xx + (0.01)*(dt/0.02)*np.random.randn(*xx.shape)
		
		allPaths.append(xx)

	return allPaths

def generate_stochastic_lorenz(n=100,T=100,dt=1,coeffs=[10,28,8/3,0.15,0.15,0.15]):

	sigma,rho,beta = coeffs[0],coeffs[1],coeffs[2]
	A1,A2,A3 = coeffs[0],coeffs[1],coeffs[2]
	t = np.arange(0,T,dt)
	assert len(t) == 40*5
	allPaths = [ ]

	for ii in range(n):

		
		xnot = np.random.randn(3)

		xx = [xnot]
		for jj in range(1,len(t)+1):

			prev = xx[jj-1]
			x = prev[0]
			y = prev[1]
			z = prev[2]

			sample_dW = np.sqrt(dt) * np.random.randn(3)
			dx = sigma * (y - x)*dt +A1 * sample_dW[0]
			dy = (x * (rho - z) - y)*dt +A2 * sample_dW[1]
			dz = (x*y  - beta*z)*dt +A3 * sample_dW[2]

			#assert (abs(dx) < 200) & (abs(dy) < 200) & (abs(dz)<200),print(dx,dy,dz,ii,jj)
			xx.append(xx[jj-1] + np.hstack([dx,dy,dz]))

		xx = np.vstack(xx)
		assert xx.shape[0] == (len(t) + 1), print(xx.shape)
		
		allPaths.append(xx)

	mu = np.nanmean(np.vstack(allPaths),axis=0)
	sd = np.nanstd(np.vstack(allPaths),axis=0)
	allPaths = [(p - mu[None,:])/sd[None,:] for p in allPaths]
	allPaths = [p + 0.01*np.random.randn(*p.shape) for p in allPaths]
	#allPaths = [p[::5] for p in allPaths]
	return allPaths

class toyDataset(Dataset):

	def __init__(self,toyData,sampledt,truedt) -> None:
		"""
		toyData: list of numpy arrays
		"""
		skip = int(sampledt/truedt)
		ourSamples = [t[::skip].squeeze() for t in toyData]
		lens = list(map(len,ourSamples))
		lens = [0] + list(np.cumsum([l for l in lens][:-1]))
		#lenall.append([l - 1 for l in lens][:-1])
		validInds = np.hstack([list(range(l,len(t)-1 + l)) \
			 for l,t in zip(lens,ourSamples)])
		print(np.hstack(ourSamples).shape)
		print(lens)
		#print(lenall)
		print(validInds)
		assert False


		self.origData = toyData
		#self.data= toyData[::skip]
		self.dt = sampledt
		self.length = self.data.shape[0] - 1
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
			result.append((self.data[ii],self.data[ii + 1],self.dt))

		if single_index:
			return result[0]
		return result
	
	def transform(self,data):
		return torch.from_numpy(data).type(torch.FloatTensor)
						
		
def makeToyDataloaders(ds1,ds2,sampledt,truedt):

	#assert ds1.shape[1] == 3
	#ds1 = ds1).type(torch.FloatTensor)
	#ds2 = torch.from_numpy(ds2).type(torch.FloatTensor)
	dataset1 = toyDataset(ds1,sampledt,truedt)
	dataset2 = toyDataset(ds2,sampledt,truedt)

	trainDataLoader = DataLoader(dataset1,batch_size=256,shuffle=True,
			      num_workers=4)
	testDataLoader = DataLoader(dataset2,batch_size=256,shuffle=False,
			      num_workers=4)
	
	return {'train':trainDataLoader,'test':testDataLoader}
