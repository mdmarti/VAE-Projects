from abc import ABC 
from abc import abstractmethod
from typing import Optional, Union, Tuple, Any
import numpy as np
import torch
import torch.nn as nn

class encoder(ABC):
	"""
	
	
	base class for all types of encoders we might implement. 
	All inheriting classes at least need a forward method,
	all else should be optional
	"""

	@abstractmethod
	def forward(
		self,
		data: torch.FloatTensor,
		pass_gradient: bool=True
	) -> torch.FloatTensor:
		
		r"""
		Return batch latent embeddings.
		Needs an option to NOT pass gradients through,
		in case that's somethign we want to do later

		Args: 
			data: what we are embedding in our latent space
			pass_gradient: whether or not we want to pass gradient
		"""

		raise NotImplementedError
	

class linearEncoder(encoder,nn.Module):

	def __init__(
			self,
			data_dim: int,
			latent_dim: int,
			has_bias: bool=True,
			device: str='cuda') -> None:
		
		super(linearEncoder,self).__init__()
		self.data_dim = data_dim
		self.latent_dim = latent_dim
		self.bias= has_bias
		self.F = nn.Linear(data_dim,latent_dim,bias=has_bias)
		self.device = device
		self.to(self.device)
		self.type = 'deterministic'

	def forward(self, 
				data: torch.FloatTensor, 
				pass_gradient: bool = True
				) -> torch.FloatTensor:
		
		if pass_gradient:
			return self.F(data)
		else:
			return self.F(data.detach())
		

class MLPEncoder(linearEncoder,nn.Module):

	def __init__(
			self,
			data_dim: int,
			latent_dim: int,
			n_hidden=0,
			hidden_size=10,
			has_bias: bool=True,
			device: str='cuda') -> None:
		
		super(MLPEncoder,self).__init__(data_dim,latent_dim,has_bias,device)
		F = []
		for _ in range(n_hidden):
			F = F + [nn.Linear(hidden_size,hidden_size),nn.Softplus()] 
		F = [nn.Linear(self.data_dim,hidden_size),nn.Softplus()] + \
						F + \
						[nn.Linear(hidden_size,self.latent_dim)]

		self.F = nn.Sequential(*F)

		self.F.to(self.device)

class ProbMLPEncoder(linearEncoder,nn.Module):

	def __init__(
			self,
			data_dim: int,
			latent_dim: int,
			n_hidden=0,
			hidden_size=10,
			has_bias: bool=True,
			device: str='cuda') -> None:
		
		
		super(ProbMLPEncoder,self).__init__(data_dim,latent_dim,has_bias,device)

		self.chol_inds = torch.tril_indices(self.latent_dim,self.latent_dim)
		self.n_entries = np.sum(list(range(1,self.latent_dim+1)))
		

		F = []
		for _ in range(n_hidden):
			F = F + [nn.Linear(hidden_size,hidden_size),nn.Softplus()] 
		F = [nn.Linear(self.data_dim,hidden_size),nn.Softplus()] + F 

		self.F = nn.Sequential(*F)
		self.mu = nn.Linear(hidden_size,self.latent_dim)
		self.D = nn.Linear(hidden_size,self.n_entries)
		self.F.to(self.device)
		self.mu.to(self.device)
		self.D.to(self.device)
		self.type = 'probabilistic'

	def forward(self, 
				data: torch.FloatTensor, 
				pass_gradient: bool = True,
				type='deterministic'
				) -> torch.FloatTensor:
		
		if pass_gradient:
			intmdt = self.F(data)
			mu = self.mu(intmdt)
			chol = torch.zeros(data.shape[0],self.latent_dim,self.latent_dim).to(self.device)
			D = torch.exp(self.D(intmdt))
			chol[:,self.chol_inds[0],self.chol_inds[1]] = D
			#return mu, chol @ chol.transpose(-2,-1)
		else:
			intmdt = self.F(data.detach())
			mu = self.mu(intmdt)
			chol = torch.zeros(data.shape[0],self.latent_dim,self.latent_dim).to(self.device)
			D = torch.exp(self.D(intmdt))
			chol[:,self.chol_inds[0],self.chol_inds[1]] = D
			
		if type == 'deterministic':
			return mu
		else:
			return mu, chol @ chol.transpose(-2,-1)


		

class ConvEncoder(linearEncoder):

	def __init__(
			self,
			data_dim: int,
			latent_dim: int,
			has_bias: bool=True,
			device:str='cuda') -> None:
		
		"""
		this is going to be hard-coded because i dont want to
		do any math if you wanna change it deal with it yourself

		YOU BREAK IT

		YOU BUY IT BADABING BADABOOM
		Just use the version from AVA

		WIP
		"""
		super(ConvEncoder,self).__init__(data_dim,latent_dim,has_bias)

		
		self.conv = nn.Sequential(*[nn.BatchNorm2d(1),
						  nn.Conv2d(1, 8, 3,1,padding=1),
						  nn.ReLU(),
						  nn.BatchNorm2d(8),
						  nn.Conv2d(8, 8, 3,2,padding=1),
						  nn.ReLU(),
						  nn.BatchNorm2d(8),
						  nn.Conv2d(8, 16,3,1,padding=1),
						  nn.ReLU(),
						  nn.BatchNorm2d(16),
						  nn.Conv2d(16,16,3,2,padding=1),
						  nn.ReLU(),
						  nn.BatchNorm2d(16),
						  nn.Conv2d(16,24,3,1,padding=1),
						  nn.ReLU(),
						  nn.BatchNorm2d(24),
						  nn.Conv2d(24,24,3,2,padding=1),
						  nn.ReLU(),
						  nn.BatchNorm2d(24),
						  nn.Conv2d(24,32,3,1,padding=1),
						  nn.ReLU()])
		self.linear = nn.Sequential(*[nn.Linear(8192,1024),
							   nn.ReLU(),
							   nn.Linear(1024,256),
							   nn.ReLU(),
							   nn.Linear(256,64),
							   nn.ReLU(),
							   nn.Linear(64,self.latent_dim)])
		
		self.device=device
		self.to(self.device)

	def forward(self, 
				data: torch.FloatTensor, 
				pass_gradient: bool = True
				) -> torch.FloatTensor:
		
		if pass_gradient:
			intmdt = self.conv(data)
			
		else:
			intmdt = self.conv(data.detach())
		
		intmdt = intmdt.view(-1,8192)
		return self.linear(intmdt)


def _softmax(x:np.array,temp=1):

	m = np.amax (x,axis=1,keepdims=True)
	e_x = np.exp((x - m)/temp)
	return e_x/np.sum(e_x,axis=1,keepdims=True) 

def _combine_dims(x:np.array):

	xOut = x 
	for ii in range(x.shape[-1]-1):
		xOut[:,ii] = x[:,ii]*x[:,ii+1]

	return xOut

def _sigmoid(x:np.array):

	
class projection:

	"""
	
	we just need one class for this
	"""

	def __init__(self,origDim,newDim, projType='linear',temp=1) -> None:
		self.d1 = origDim
		self.d2 = newDim
		self.projType=projType 
		self.W = np.random.randn(origDim,newDim)
		self.temp=temp
		if projType == 'linear':
			

			self.projection = lambda x: x @ self.W 
		
		elif projType == 'softmax':

			assert self.temp >0, print('Temperature should be a positive number')
			self.projection = lambda x: _softmax(x @ self.W,self.temp)

		elif projType == 'combine':

			self.projection = lambda x: _combine_dims(x @ self.W)
		else:
			print('Method must be softmax or linear')
			raise NotImplementedError

	
	def project(
		self,
		data: np.array,
		noise: float= 0.
	 ) -> np.array:
		
		r"""
		Return batch projectiojns.
		

		Args: 
			data: what we are embedding in a higher-d space
			
		"""
		if noise:
			proj = self.projection(data)
			return  proj + noise * np.random.normal(size=proj.shape)
		else:
			return self.projection(data)
	

