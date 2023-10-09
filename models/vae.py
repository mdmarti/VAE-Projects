import torch
from encoding import *
from decoding import *
import torch.nn as nn
from torch.distributions import LowRankMultivariateNormal
from torch.utils.tensorboard import SummaryWriter
import os

class VAE(nn.Module):

	def __init__(self,latent_dim,data_shape,encoder,decoder,save_dir='',
				 model_precision=1e-3,device='cuda'):
		  
		super(VAE,self).__init__()
		self.save_dir=save_dir
		self.latent_dim=latent_dim
		self.data_shape = data_shape
		self.data_dim = np.prod(data_shape)
		self.precision=model_precision 
		
		self.encoder = encoder 
		self.decoder = decoder 

		self.device = device 
		self.writer = SummaryWriter(log_dir = os.path.join(self.save_dir,'runs'))
		self.epoch=0
		self.to(self.device)


	def loss(self,x,return_latent_rec=False):

		mu,u,d = self.encoder.forward(x)
		latent_dist = LowRankMultivariateNormal(mu,u,d)
		z = latent_dist.rsample
		x_rec = self.decoder.forward(z)

		elbo = -0.5 * (torch.sum(torch.pow(z,2)) + self.z_dim * np.log(2*np.pi))
		# E_{q(z|x)} p(x|z)
		pxz_term = -0.5 * self.data_dim * (np.log(2*np.pi/self.model_precision))
		l2s = torch.sum(torch.pow(x.view(x.shape[0],-1) - x_rec, 2), dim=1)
		pxz_term = pxz_term - 0.5 * self.model_precision * torch.sum(l2s)
		elbo = elbo + pxz_term
		# H[q(z|x)]
		elbo = elbo + torch.sum(latent_dist.entropy())
		if return_latent_rec:
			return -elbo, z.detach().cpu().numpy(), \
				x_rec.view(-1, self.data_shape[0], self.data_shape[1]).detach().cpu().numpy()
		return -elbo

	def forward(self,batch,return_latent_rec=False):
		

		x,_,_ = batch 
		x = x.to(self.device)

		return self.loss(x,return_latent_rec)
	

	def train_epoch(self,loader,optimizer,grad_clipper=None):

		self.train()
		epoch_loss = 0. 
		for ii,batch in enumerate(loader):
			optimizer.zero_grad()
			loss = self.forward(batch)
			epoch_loss += loss.item()

			loss.backward()
			if grad_clipper != None:

				grad_clipper(self.parameters())

			optimizer.step()

		self.writer.add_scalar('Train/loss',epoch_loss/len(loader),self.epoch)


	def test_epoch(self,loader):

		self.eval()
		with torch.no_grad():
			epoch_loss = 0.
			for batch in loader:
				loss = self.forward(batch)
				
				epoch_loss += loss.item()

		self.writer.add_scalar('Test/loss',epoch_loss/len(loader),self.epoch)
		
		return epoch_loss
	
	def save(self):

		sd = self.state_dict()
		fn = os.path.join(self.save_dir,f'checkpoint_{self.epoch}.tar')
		torch.save({
			'epoch': self.epoch,
			'model_state_dict':sd
		},fn)
		return 
	
	def load(self,fn):
		print(f"Loading state from: {fn}")
		check = torch.load(fn,map_location=self.device)
		self.epoch = check['epoch']
		self.load_state_dict(check['model_state_dict'])
		return
	
	def encode_trajectory(self,data):

		return self.encoder.forward(data.to(self.device)).detach().cpu().numpy()

			


