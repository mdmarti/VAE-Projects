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
	

class linearEncoder(encoder):

	def __init__(
			self,
			data_dim: int,
			latent_dim: int,
			has_bias: bool=True) -> None:
		
		self.data_dim = data_dim
		self.latent_dim = latent_dim
		self.bias= has_bias
		self.F = nn.Linear(data_dim,latent_dim,bias=has_bias)

	def forward(self, 
				data: torch.FloatTensor, 
				pass_gradient: bool = True
				) -> torch.FloatTensor:
		
		if pass_gradient:
			return self.F(data)
		else:
			return self.F(data.detach())
		

class MLPEncoder(linearEncoder):

	def __init__(
			self,
			data_dim: int,
			latent_dim: int,
			n_hidden=0,
			hidden_size=10,
			has_bias: bool=True) -> None:
		
		self.data_dim = data_dim
		self.latent_dim = latent_dim
		self.bias= has_bias
		F = []
		for _ in range(n_hidden):
			F = F + [nn.Linear(hidden_size,hidden_size),nn.Softplus()] 
		F = [nn.Linear(self.data_dim,hidden_size),nn.Softplus()] + \
						F + \
						[nn.Linear(hidden_size,self.latent_dim)]

		self.F = nn.Sequential(*F)

class ConvEncoder(linearEncoder):

	def __init__(
			self,
			data_dim: int,
			latent_dim: int,
			n_conv=5,
			hidden_size=10,
			has_bias: bool=True) -> None:
		
		"""
		this is going to be hard-coded because i dont want to
		do any math if you wanna change it deal with it yourself

		YOU BREAK IT

		YOU BUY IT BADABING BADABOOM
		Just use the version from AVA

		WIP
		"""
		self.data_dim = data_dim
		self.latent_dim = latent_dim
		self.bias= has_bias
		F = [nn.Conv2d(in_channels=1,out_channels=4)]



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
	 ) -> np.array:
		
		r"""
		Return batch projectiojns.
		

		Args: 
			data: what we are embedding in a higher-d space
			
		"""

		return self.projection(data)
	

