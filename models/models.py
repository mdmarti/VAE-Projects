import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal, MultivariateNormal, kl_divergence,Laplace
import os
import numpy as np


class encoder(nn.Module):

	def __init__(self,z_dim):

		"""
		encoder for birdsong VAEs
		"""

		super(encoder,self).__init__()

		self.z_dim = z_dim

		self.encoder_conv = nn.Sequential(nn.BatchNorm2d(1),
								nn.Conv2d(1, 8, 3,1,padding=1),
								nn.Relu(),
								nn.BatchNorm2d(8),
								nn.Conv2d(8, 8, 3,2,padding=1),
								nn.Relu(),
								nn.BatchNorm2d(8),
								nn.Conv2d(8, 16,3,1,padding=1),
								nn.Relu(),
								nn.BatchNorm2d(16),
								nn.Conv2d(16,16,3,2,padding=1),
								nn.Relu(),
								nn.BatchNorm2d(16),
								nn.Conv2d(16,24,3,1,padding=1),
								nn.Relu(),
								nn.BatchNorm2d(24),
								nn.Conv2d(24,24,3,2,padding=1),
								nn.Relu(),
								nn.BatchNorm2d(24),
								nn.Conv2d(24,32,3,1,padding=1),
								nn.Relu())

		self.encoder_fc = nn.Sequential(nn.Linear(8193,1024),
								nn.Relu(),
								nn.Linear(1024,256),
								nn.Relu())

		self.fc11 = nn.Linear(256,64)
		self.fc12 = nn.Linear(256,64)
		self.fc13 = nn.Linear(256,64)
		self.fc21 = nn.Linear(64,self.z_dim)
		self.fc22 = nn.Linear(64,self.z_dim)
		self.fc23 = nn.Linear(64,self.z_dim)
		
	def encode(self,x):

		h = self.encoder_conv(x)
		h = h.view(-1,8192)
		h = torch.cat(h,torch.zeros(h.shape[0],1,device=h.device))
		h = self.encoder_fc(x)
		mu = F.relu(self.fc11(h))
		u = F.relu(self.fc12(h))
		d = F.relu(self.fc13(h))
		mu = self.fc21(h)
		u = self.fc22(h)
		d = self.fc23(h)
		
		return mu, u.unsqueeze(-1),d.exp()

	def encode_with_time(self,x,encode_times):

		h = self.encoder_conv(x)
		h = h.view(-1,8192)
		h = torch.cat(h,encode_times)
		h = self.encoder_fc(x)
		mu = F.relu(self.fc11(h))
		u = F.relu(self.fc12(h))
		d = F.relu(self.fc13(h))
		mu = self.fc21(h)
		u = self.fc22(h)
		d = self.fc23(h)
		
		return mu, u.unsqueeze(-1),d.exp()

	def sample_z(self,mu,u,d):

		dist = LowRankMultivariateNormal(mu,u,d)

		z_hat = dist.rsample()

		return z_hat


class decoder(nn.module):

	def __init__(self, z_dim=32,precision =1e4):
		"""
		Initialize stupid decoder

		Inputs
		-----
			z_dim: int, dim of latent dimension
			x_dim: int, dim of input data
			decoder_dist: bool, determines if we learn var of decoder in addition
						to mean
		"""

		super(decoder,self).__init__()
		self.precision = precision
		self.decoder_fc = nn.Sequential(nn.Linear(z_dim,64),
										nn.Linear(64,256),
										nn.Linear(256,1024),
										nn.Relu(),
										nn.Linear(1024,8193),
										nn.Relu())
		self.decoder_convt = nn.Sequential(nn.BatchNorm2d(32),
										nn.ConvTranspose2d(32,24,3,1,padding=1),
										nn.Relu(),
										nn.BatchNorm2d(24),
										nn.ConvTranspose2d(24,24,3,2,padding=1,output_padding=1),
										nn.Relu(),
										nn.BatchNorm2d(24),
										nn.ConvTranspose2d(24,16,3,1,padding=1),
										nn.Relu(),
										nn.BatchNorm2d(16),
										nn.ConvTranspose2d(16,16,3,2,padding=1,output_padding=1),
										nn.Relu(),
										nn.BatchNorm2d(16),
										nn.ConvTranspose2d(16,8,3,1,padding=1),
										nn.Relu(),
										nn.BatchNorm2d(8),
										nn.ConvTranspose2d(8,8,3,2,padding=1,output_padding=1),
										nn.Relu(),
										nn.BatchNorm2d(8),
										nn.ConvTranspose2d(8,1,3,1,padding=1))


		#device_name = "cuda" if torch.cuda.is_available() else "cpu"
		#self.device = torch.device(device_name)

		#self.to(self.device)


	def decode(self,z,return_time = False):
		"""
		Decode latent samples

		Inputs
		-----
			z: torch.tensor, latent samples to be decoded

		Outputs
		-----
			mu: torch.tensor, mean of decoded distribution
			if decoder_dist:
				logvar: torch.tensor, logvar of decoded distribution
		"""
		#print(z.dtype)
		z = self.decoder_fc(z)
		that = z[:,-1]
		z = z[:,:-1]
		z = z.view(-1,32,16,16)
		xhat = self.decoder_convt(z)
		#mu = self.mu_convt(z)

		if return_time:
			return xhat, that
		else:
			return xhat

class VAE_Base(nn.module):

	def __init__(self, encoder, decoder,save_dir,lr=1e-4):

		super(VAE_Base,self).init()

		self.encoder = encoder 
		self.decoder = decoder 
		self.z_dim = self.encoder.z_dim
		self.save_dir = save_dir 
		self.epoch = 0
		self.lr = lr

		device_name = "cuda" if torch.cuda.is_available() else "cpu"
		self.device = torch.device(device_name)
		self.to(self.device)
		self.optimizer = Adam(self.parameters(), lr=lr) # 8e-5


	def _compute_reconstruction_loss(self,x,xhat):

		constant = -0.5 * np.prod(x.shape[1:]) * np.log(2*np.pi/self.vae.precision)

		x = torch.reshape(x,(x.shape[0],-1))
		xhat = torch.reshape(xhat,(xhat.shape[0],-1))
		l2s = torch.sum(torch.pow(x - mean,2),axis=1)

		logprob = constant - 0.5 * self.vae.precision * torch.sum(l2s)

		return logprob

	def _compute_kl_loss(self,mu,u,d):

		## uses matrix determinant lemma to compute logdet of covar
		Ainv = torch.diag_embed(1/d)
		term1 = torch.log(1 + u.T @ Ainv @ u)
		term2 = torch.log(d).sum()

		ld = term1 + term2 

		mean = torch.pow(mu,2)

		trace = torch.diag(u @ u.T + 1/Ainv)

		kl = 0.5 * ((trace + mean - 1).sum(axis=1) - ld)

		return kl


	def compute_loss(self,x,return_recon = False):


		mu,u,d = self.encoder.encode(x)

		dist = LowRankMultivariateNormal(mu,u,d)

		zhat = dist.rsample()

		xhat = self.decoder.decode(zhat)

		kl = self._compute_kl_loss(mu,u,d)
		logprob = self._compute_reconstruction_loss(x,xhat)

		elbo = logprob - kl 

		if return_recon:
			return -elbo,logprob, kl, xhat.view(-1,128,128).detach().cpu().numpy() 
		else:
			return -elbo,logprob, kl


	def train_epoch(self,train_loader):

		self.train()

		train_loss = 0.0
		train_kl = 0.0
		train_lp = 0.0 

		for ind, batch in enumerate(train_loader):

			self.optimizer.zero_grad()
			(spec,day) = batch 
			day = day.to(self.device)

			spec = torch.stack(spec,axis=0)
			spec = spec.to(self.device)

			loss,lp,kl = self.compute_loss(spec)

			train_loss += loss.item()
			train_kl += kl.item()
			train_lp += lp.item()

			loss.backward()
			self.optimizer.step()

		train_loss /= len(train_loader)
		train_kl /= len(train_loader)
		train_lp /= len(train_loader)

		print('Epoch {0:d} average train loss: {1:.3f}'.format(self.epoch,train_loss))
		print('Epoch {0:d} average train kl: {1:.3f}'.format(self.epoch,train_kl))
		print('Epoch {0:d} average train lp: {1:.3f}'.format(self.epoch,train_lp))

		return train_loss

	def test_epoch(self,test_loader):

		self.eval()
		test_loss = 0.0
		test_kl = 0.0
		test_lp = 0.0 

		for ind, batch in enumerate(test_loader):

			(spec,day) = batch 
			day = day.to(self.device)

			spec = torch.stack(spec,axis=0)
			spec = spec.to(self.device)
			with torch.no_grad():
				loss,lp,kl = self.compute_loss(spec)

			test_loss += loss.item()
			test_kl += kl.item()
			test_lp += lp.item()

			
		test_loss /= len(test_loader)
		test_kl /= len(test_loader)
		test_lp /= len(test_loader)

		print('Epoch {0:d} average test loss: {1:.3f}'.format(self.epoch,test_loss))
		print('Epoch {0:d} average test kl: {1:.3f}'.format(self.epoch,test_kl))
		print('Epoch {0:d} average test lp: {1:.3f}'.format(self.epoch,test_lp))

		return test_loss

	def visualize(self,loader,n_recons=10):


	def get_latent(self,loader):

		latents = []

		for ind, batch in enumerate(loader):

			(spec,day) = batch 
			day = day.to(self.device)

			spec = torch.stack(spec,axis=0)
			with torch.no_grad():
				z_mu,_,_ = self.encoder.encode(spec)

			latents.append(z_mu.detach().cpu().numpy())

		return latents

	def save_state(self):

		"""
		Save state of network. Saves encoder and decoder state separately. Assumes
		that while training, you have been using set_epoch to set the epoch
		"""

		#self.set_epoch(epoch)
		fn = os.path.join(self.save_dir, 'checkpoint_encoder_' + str(self.epoch) + '.tar')

		"""Save state."""
		sd = self.state_dict()
		torch.save({
				'model_state_dict': sd,
				'optimizer_state_dict': self.optimizer.state_dict(),
				'epoch': self.epoch
			}, fn)

	def load_state(self,fn):

		"""
		Load state of network. Requires an epoch to recover the current state of network

		Inputs:
		-----
			epoch: int, current epoch of training
		"""
		"""Load state."""

		print("Loading state from:", fn)
		#print(self.state_dict().keys())

		checkpoint = torch.load(fn, map_location=self.device)
		layer_1 = checkpoint['model_state_dict'].pop('layer_1')

		self.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		self.epoch = checkpoint['epoch']

	

