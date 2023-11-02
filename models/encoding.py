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
						[nn.Linear(hidden_size,self.latent_dim),nn.BatchNorm1d(self.latent_dim)]

		self.F = nn.Sequential(*F)

		self.to(self.device)

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
							   nn.Linear(64,self.latent_dim),
							   nn.BatchNorm1d(self.latent_dim)])
		
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


def _softmax(x:np.array):

	m = np.amax (x,axis=1)
	e_x = np.exp(x - m)
	return e_x/np.sum(e_x,axis=1) 

class projection:

	"""
	
	we just need one class for this
	"""

	def __init__(self,origDim,newDim, projType='linear') -> None:
		self.d1 = origDim
		self.d2 = newDim
		self.projType=projType 
		self.W = np.random.randn(origDim,newDim)
		if projType == 'linear':
			

			self.projection = lambda x: x @ self.W 
		
		elif projType == 'softmax':

			self.projection = lambda x: _softmax(x @ self.W)

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
	

