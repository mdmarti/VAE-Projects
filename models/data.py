import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

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
	for ii in range(n):

		xnot = x0 + 0.03**2 * np.random.randn(1)
		xx = [xnot]

		for jj in range(1,len(t)+1):
			x = xx[jj-1]
			sample_dW = np.sqrt(dt) * np.random.randn(1)
			dx = mu*x *dt + sigma*x * sample_dW
			xx.append(xx[jj-1] + dx)
		xx = np.hstack(xx)[:,None]

		#xx = xx + (0.01)*(dt/0.02)*np.random.randn(*xx.shape)
		
		allPaths.append(xx)

	mu_fnc = lambda x: mu * x.detach().cpu().numpy() 
	d_fnc = lambda x: sigma * x.detach().cpu().numpy()

	eta_fnc = lambda x: mu_fnc(x)/d_fnc(x)**2
	lam_fnc = lambda x: 1/d_fnc(x)

	return allPaths,(mu_fnc,d_fnc),(eta_fnc,lam_fnc)

def generate_2d_swirls(n=100,T=1,dt=0.001,theta=10,mu=1.01,sigma=0.5,x0=np.array([0.1,0.1])):

	allPaths=[]
	t = np.arange(0,T,dt)
	#print("not adding noise")
	R = np.array([[np.cos(theta*np.pi/180),-np.sin(theta*np.pi/180)],\
		[np.sin(theta*np.pi/180),np.cos(theta*np.pi/180)]])
	A = mu*(R - np.eye(2))
	evals,_ = np.linalg.eig(A)
	print(evals)
	for ii in range(n):

		xnot = x0 + 0.03**2 * np.random.randn(2)
		xx = [xnot]

		

		for jj in range(1,len(t)+1):
			x = xx[jj-1]
			sample_dW = np.sqrt(dt) * np.random.randn(2)
			dx = A @ x + sigma*x * sample_dW
			xx.append(xx[jj-1] + dx)
		xx = np.vstack(xx)
		#xx = xx + (0.01)*(dt/0.02)*np.random.randn(*xx.shape)
		
		allPaths.append(xx)

	sigMat = np.eye(2) * sigma
	LMat = np.eye(2) / sigma
	inds = np.tril_indices(2)
	mu_fnc = lambda x: (A @ x.detach().cpu().numpy()[:,:,None]).squeeze()
	d_fnc = lambda x: torch.diag_embed(sigma * x).detach().cpu().numpy()[:,inds[0],inds[1]]

	eta_fnc = lambda x: ((LMat @ x[:,:,None].detach().cpu().numpy())**2 * (A @ x[:,:,None].detach().cpu().numpy())).squeeze()
	lam_fnc = lambda x: (torch.diag_embed(1/(sigma * x))).detach().cpu().numpy()[:,inds[0],inds[1]]
	return allPaths,(mu_fnc,d_fnc),(eta_fnc,lam_fnc)

def generate_stochastic_lorenz(n=100,T=1,dt=0.001,coeffs=[10,28,8/3,0.15,0.15,0.15],zscore=False):

	sigma,rho,beta = coeffs[0],coeffs[1],coeffs[2]
	A = np.array([coeffs[0],coeffs[1],coeffs[2]])
	Sig = np.eye(3) * A
	t = np.arange(0,T,dt)
	
	allPaths = [ ]

	for ii in range(n):

		
		xnot = np.random.randn(3)

		xx = [xnot]
		for jj in range(1,len(t)+1):

			prev = xx[jj-1]
			x = prev[0]
			y = prev[1]
			z = prev[2]

			sample_dW = A @ (np.sqrt(dt) * np.random.randn(3))
			dx = sigma * (y - x)*dt #+ sample_dW[0]
			dy = (x * (rho - z) - y)*dt #+ sample_dW[1]
			dz = (x*y  - beta*z)*dt #+ sample_dW[2]

			#assert (abs(dx) < 200) & (abs(dy) < 200) & (abs(dz)<200),print(dx,dy,dz,ii,jj)
			xx.append(xx[jj-1] + np.hstack([dx,dy,dz]) + sample_dW)

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
	d_fnc = lambda x: np.sqrt(dt)*torch.diag_embed(torch.FloatTensor([A1,A2,A3]))[None,:,:].repeat(x.shape[0],1,1).detach().cpu().numpy()[:,inds[0],inds[1]]

	eta_fnc = lambda x: np.vstack([(sigma * (x[:,1] - x[:,0])).detach().cpu().numpy() /A1**2,
										(x[:,0]*(rho - x[:,2]) - x[:,1]).detach().cpu().numpy() /A2**2,
										(x[:,0]*x[:,1] - beta*x[:,2]).detach().cpu().numpy() /A3**2]).T
	lam_fnc = lambda x: torch.diag_embed(torch.FloatTensor([1/A1,1/A2,1/A3]))[None,:,:].repeat(x.shape[0],1,1).detach().cpu().numpy()[:,inds[0],inds[1]]/np.sqrt(dt)
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
