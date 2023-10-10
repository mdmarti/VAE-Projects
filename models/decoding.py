import torch.nn as nn
from abc import ABC 
from abc import abstractmethod
import torch 


class decoder(ABC):
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
		Return batch decodings.
		Needs an option to NOT pass gradients through,
		in case that's somethign we want to do later

		Args: 
			data: what we are embedding in our latent space
			pass_gradient: whether or not we want to pass gradient
		"""

		raise NotImplementedError
	

class linearDecoder(decoder,nn.Module):

	def __init__(
			self,
			data_dim: int,
			latent_dim: int,
			has_bias: bool=True,
			device: str='cuda') -> None:
		
		super(linearDecoder,self).__init__()
		self.data_dim = data_dim
		self.latent_dim = latent_dim
		self.bias= has_bias
		self.F = nn.Linear(latent_dim,data_dim,bias=has_bias)
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
		

class MLPDecoder(linearDecoder,nn.Module):

	def __init__(
			self,
			data_dim: int,
			latent_dim: int,
			n_hidden=0,
			hidden_size=10,
			has_bias: bool=True,
			device: str='cuda') -> None:
		
		super(linearDecoder,self).__init__(data_dim,latent_dim,has_bias,device)
		F = []
		for _ in range(n_hidden):
			F = F + [nn.Linear(hidden_size,hidden_size),nn.Softplus()] 
		F = [nn.Linear(self.latent,hidden_size),nn.Softplus()] + \
						F + \
						[nn.Linear(hidden_size,self.data_dim)]

		self.F = nn.Sequential(*F)
		
		self.to(self.device)

class ConvDecoder(linearDecoder,nn.Module):

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
		super(ConvDecoder,self).__init__(data_dim,latent_dim,has_bias)

		
		self.conv = nn.Sequential([nn.BatchNorm2d(32),
						  nn.ConvTranspose2d(32, 24,3,1,padding=1),
						  nn.ReLU(),
						  nn.BatchNorm2d(24),
						  nn.ConvTranspose2d(24, 24, 3,2,padding=1,output_padding=1),
						  nn.ReLU(),
						  nn.BatchNorm2d(24),
						  nn.Conv2d(24, 16,3,1,padding=1),
						  nn.ReLU(),
						  nn.BatchNorm2d(16),
						  nn.Conv2d(16,16,3,2,padding=1,output_padding=1),
						  nn.ReLU(),
						  nn.BatchNorm2d(16),
						  nn.Conv2d(16,8,3,1,padding=1),
						  nn.ReLU(),
						  nn.BatchNorm2d(8),
						  nn.Conv2d(8,8,3,2,padding=1,output_padding=1),
						  nn.ReLU(),
						  nn.BatchNorm2d(8),
						  nn.Conv2d(8,1,3,1,padding=1)])
		self.linear = nn.Sequential([nn.Linear(self.z_dim,64),
							   nn.ReLU(),
                               nn.Linear(64,256),
							   nn.ReLU(),
                               nn.Linear(256,1024),
							   nn.ReLU(),
                               nn.Linear(1024,8192),
							   nn.ReLU()])
		self.device=device
		self.to(self.device)

	def forward(self, 
				data: torch.FloatTensor, 
				pass_gradient: bool = True
				) -> torch.FloatTensor:
		
		if pass_gradient:
			intmdt = self.linear(data)
			
		else:
			intmdt = self.linear(data.detach())
		
		intmdt = intmdt.view(-1,32,16,16)
		return self.conv(intmdt)