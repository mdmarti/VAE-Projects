import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import inspect
import sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir) 
import copy
from VAE_Projects.experiments.plotting import plot_trajectories_umap_and_coords


class resnet_encoder(nn.Module):

	def __init__(self):

		"""
		resnet encoder
		"""
		super().__init__()
		self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
		self.f = nn.Sequential(nn.BatchNorm1d(1000),
								nn.Linear(1000,2048),
								nn.ReLU(),
								nn.BatchNorm1d(2048),
								nn.Linear(2048,2048),
								nn.ReLU(),
								nn.BatchNorm1d(2048),
								nn.Linear(2048,2048))
		
	def encode(self,x):

		z = self.model(x)
		z = z.view(-1,8192)

		z = self.f(z)
		
		return z

class resnet_predictor(nn.Module):

	def __init__(self) -> None:
		super(resnet_predictor,self).__init__()	
	
		self.h = nn.Sequential(nn.BatchNorm1d(2048),
								nn.Linear(2048,512),
								nn.ReLU(),
								nn.Linear(512,2048))

	def predict(self, z):

		p = self.h(z)

		return p 

class encoder(nn.Module):

	def __init__(self,z_dim=128):

		"""
		encoder for birdsong VAEs
		"""

		super(encoder,self).__init__()

		self.z_dim = z_dim

		self.encoder_conv = nn.Sequential(nn.BatchNorm2d(1),
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
								nn.ReLU())

		self.encoder_fc = nn.Sequential(nn.Linear(8192,1024),
								nn.ReLU(),
								nn.Linear(1024,self.z_dim),
								nn.ReLU())

		#self.fc11 = nn.Linear(256,64)
		#self.fc12 = nn.Linear(256,64)
		#self.fc13 = nn.Linear(256,64)
		#self.fc21 = nn.Linear(64,self.z_dim)
		#self.fc22 = nn.Linear(64,self.z_dim)
		#self.fc23 = nn.Linear(64,self.z_dim)
		
	def encode(self,x):

		z = self.encoder_conv(x)
		z = z.view(-1,8192)

		z = self.encoder_fc(z)
		
		return z

class predictor(nn.Module):

	def __init__(self,z_dim=128,h_dim=64):

		"""
		encoder for birdsong VAEs
		"""

		super(predictor,self).__init__()
		
		self.z_dim = z_dim 
		self.h_dim = h_dim 

		self.predictor_fc = nn.Sequential(nn.Linear(self.z_dim,self.h_dim),
									nn.ReLU(),
									nn.Linear(self.h_dim,self.z_dim),
									nn.ReLU())

	def predict(self, z):

		p = self.predictor_fc(z)

		return p 


class simsiam(nn.Module):


	def __init__(self,encoder=None,predictor=None,sim_func=None,save_dir='',lr=1e-4,wd=True):

		"""
		simsiam for birdsong VAEs
		"""

		super(simsiam,self).__init__()

		self.encoder=encoder 
		self.predictor=predictor
		self.sim_func=sim_func

		self.save_dir = save_dir 
		self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir,'runs'))

		self.epoch = 0
		self.lr = lr
		self.loss = {'train': {}, 'test': {}}

		device_name = "cuda" if torch.cuda.is_available() else "cpu"
		self.device = torch.device(device_name)
		self.to(self.device)
		self.wd = wd 
		if self.wd:
			print('gonna decay my weights')
			self.optimizer = Adam(self.parameters(),lr=lr, weight_decay=1e-2)
		else:
			self.optimizer = Adam(self.parameters(), lr=lr) # 8e-5


	def encode(self,x):

		z = self.encoder.encode(x)

		p = self.predictor.predict(z)

		return z,p

	def compute_loss(self,z,p):

		z_loss = z.detach()
		l = self.sim_func(p,z_loss)

		return l 

	def train_epoch(self, loader):

		self.train()
		train_loss = 0.0
		#loader.dataset.train_augment=True
		for ii, batch in enumerate(loader):

			(x1,x2) = batch
			
			x1,x2 = x1.unsqueeze(1).to(self.device),x2.unsqueeze(1).to(self.device)

			
			z1,p1 = self.encode(x1)
			z2,p2 = self.encode(x2)

			L = self.compute_loss(z1,p2)/2 + self.compute_loss(z2,p1)/2

			train_loss += L.item()
			
			L.backward()
			self.optimizer.step()

		train_loss /= len(loader)

		print('Epoch {0:d} average train loss: {1:.3f}'.format(self.epoch,train_loss))

		return train_loss 

	def test_epoch(self,loader):

		self.eval()

		#loader.dataset.train_augment=True
		test_loss = 0.0

		for ii,batch in enumerate(loader):

			(x1,x2) = batch
			
			x1,x2 = x1.unsqueeze(1).to(self.device),x2.unsqueeze(1).to(self.device)

			with torch.no_grad():
				z1,p1 = self.encode(x1)
				z2,p2 = self.encode(x2)

				L = self.compute_loss(z1,p2)/2 + self.compute_loss(z2,p1)/2

				test_loss += L.item()

		test_loss /= len(loader)

		print('Epoch {0:d} average test loss: {1:.3f}'.format(self.epoch,test_loss))

		return test_loss 


	def train_test_loop(self,loaders, epochs=100, test_freq=2, save_freq=10,vis_freq=10):
		"""
		Train the model for multiple epochs, testing and saving along the way.

		Parameters
		----------
		loaders : dictionary
			Dictionary mapping the keys ``'test'`` and ``'train'`` to respective
			torch.utils.data.Dataloader objects.
		epochs : int, optional
			Number of (possibly additional) epochs to train the model for.
			Defaults to ``100``.
		test_freq : int, optional
			Testing is performed every `test_freq` epochs. Defaults to ``2``.
		save_freq : int, optional
			The model is saved every `save_freq` epochs. Defaults to ``10``.
		vis_freq : int, optional
			Syllable reconstructions are plotted every `vis_freq` epochs.
			Defaults to ``1``.
		"""
		print("="*40)
		print("Training: epochs", self.epoch, "to", self.epoch+epochs-1)
		print("Training set:", len(loaders['train'].dataset))
		print("Test set:", len(loaders['test'].dataset))
		print("="*40)
		# For some number of epochs...
		for epoch in range(self.epoch, self.epoch+epochs):
			# Run through the training data and record a loss.
			loss = self.train_epoch(loaders['train'])
			self.loss['train'][epoch] = loss
			# Run through the test data and record a loss.
			if (test_freq is not None) and (epoch % test_freq == 0):
				loss = self.test_epoch(loaders['test'])
				self.loss['test'][epoch] = loss
			# Save the model.
			if (save_freq is not None) and (epoch % save_freq == 0) and \
					(epoch > 0):
				filename = "checkpoint_"+str(epoch).zfill(3)+'.tar'
				self.save_state()
			if (vis_freq is not None) and (epoch % vis_freq == 0):

				filename = "epoch_"+str(epoch) + '_'

				plot_trajectories_umap_and_coords(self,copy.deepcopy(loaders['test']),save_prefix=filename)
			self.epoch += 1

	def get_latent(self,loader):

		'''
		this requires a loader that does NOT augment images beforehand
		'''
		latents = []
		#print(len(loader))#.dataset.train_augment)
		tmpdl = DataLoader(loader.dataset, batch_size=1, \
			shuffle=False, num_workers=loader.num_workers)
		tmpdl.dataset.train_augment=False
		#print(len(loader))
		#assert False
		
		for ind, batch in enumerate(tmpdl):

			(x1,_) = batch 
			

			x1 = torch.vstack(x1).unsqueeze(1).to(self.device)
			with torch.no_grad():
				z = self.encoder.encode(x1)

			latents.append(z.detach().cpu().numpy())

		
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
		#layer_1 = checkpoint['model_state_dict'].pop('layer_1')

		self.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		self.epoch = checkpoint['epoch']

	def latents_plots(self,loader=None,n_samples = 10):

		samples = np.random.choice(len(loader),n_samples)

		for s in samples:
			loader.dataset

		return


class normed_simsiam(simsiam):

	def __init__(self,encoder=None,predictor=None,sim_func=None,save_dir='',lr=1e-4,wd=False):

		"""
		simsiam for birdsong VAEs
		"""

		super(normed_simsiam,self).__init__(encoder=encoder,predictor=predictor,sim_func=sim_func,save_dir=save_dir,lr=lr,wd=wd)

	def encode(self,x):

		z = self.encoder.encode(x)
		
		z = z/torch.norm(z,dim=-1,keepdim=True)

		p = self.predictor.predict(z)

		p = p/torch.norm(p,dim=-1,keepdim=True)

		return z,p
	
	def get_latent(self,loader):

		'''
		this requires a loader that does NOT augment images beforehand
		'''
		latents = []
		#print(len(loader))#.dataset.train_augment)
		tmpdl = DataLoader(loader.dataset, batch_size=1, \
			shuffle=False, num_workers=loader.num_workers)
		tmpdl.dataset.train_augment=False
		#print(len(loader))
		#assert False
		
		for ind, batch in enumerate(tmpdl):

			(x1,_) = batch 
			

			x1 = torch.vstack(x1).unsqueeze(1).to(self.device)
			with torch.no_grad():
				z = self.encoder.encode(x1)
				#z = z/torch.norm(z,dim=-1,keepdim=True)

			latents.append(z.detach().cpu().numpy())

		
		return latents


class image_simsiam(simsiam):

	def __init__(self, encoder=None, predictor=None, sim_func=None, save_dir='', lr=0.0001,transforms=()):
		super(image_simsiam,self).__init__(encoder, predictor, sim_func, save_dir, lr)

		self.transforms=transforms

	def apply_transforms(self,x):

		tx = self.transforms(x)

		return tx

	def train_epoch(self, loader):

		self.train()
		train_loss = 0.0
		loader.dataset.train_augment=True
		for ii, batch in enumerate(loader):

			(x1,x2) = batch
			
			x1,x2 = x1.unsqueeze(1).to(self.device),x2.unsqueeze(1).to(self.device)

			x1,x2 = self.apply_transforms(x1),self.apply_transforms(x2)
			z1,p1 = self.encode(x1)
			z2,p2 = self.encode(x2)

			L = self.compute_loss(z1,p2)/2 + self.compute_loss(z2,p1)/2

			train_loss += L.item()
			
			L.backward()
			self.optimizer.step()

		train_loss/len(loader)

		print('Epoch {0:d} average train loss: {1:.3f}'.format(self.epoch,train_loss))

		return train_loss 

	def test_epoch(self,loader):

		self.eval()

		loader.dataset.train_augment=True
		test_loss = 0.0

		for ii,batch in enumerate(loader):

			(x1,x2) = batch
			
			x1,x2 = x1.unsqueeze(1).to(self.device),x2.unsqueeze(1).to(self.device)

			with torch.no_grad():
				x1,x2 = self.apply_transforms(x1),self.apply_transforms(x2)
				z1,p1 = self.encode(x1)
				z2,p2 = self.encode(x2)

				L = self.compute_loss(z1,p2)/2 + self.compute_loss(z2,p1)/2

				test_loss += L.item()

		test_loss /= len(loader)

		print('Epoch {0:d} average test loss: {1:.3f}'.format(self.epoch,test_loss))

		return test_loss 


class simsiam_l1(simsiam):

	def __init__(self,encoder=None,predictor=None,sim_func=None,save_dir='',lr=1e-4,lamb=1e-10):

		"""
		simsiam for birdsong VAEs
		"""

		super(simsiam_l1,self).__init__(encoder=encoder,predictor=predictor,sim_func=sim_func,save_dir=save_dir,lr=lr)
		self.lamb = lamb
		self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir,'runs'))

	def compute_loss(self,z,p):

		z_loss = z.detach()
		l = self.sim_func(p,z_loss)

		l1_term_z = torch.abs(z_loss).sum(axis=1).mean()
		l1_term_p = torch.abs(p).sum(axis=1).mean()

		return l, l1_term_z, l1_term_p


	def train_epoch(self, loader):

		self.train()
		train_loss = 0.0
		train_sim = 0.0
		train_l1_z = 0.0
		train_l1_p = 0.0
		#loader.dataset.train_augment=True
		for ii, batch in enumerate(loader):

			(x1,x2) = batch
			
			x1,x2 = x1.unsqueeze(1).to(self.device),x2.unsqueeze(1).to(self.device)

			
			z1,p1 = self.encode(x1)
			z2,p2 = self.encode(x2)

			sim1,l1_z1,l1_p1 = self.compute_loss(z1,p2)
			sim2,l1_z2,l1_p2 = self.compute_loss(z2,p1)
			sim = (sim1 + sim2)/2
			l1_z = (l1_z1 + l1_z2)/2
			l1_p = (l1_p1 + l1_p2)/2

			L = sim + self.lamb * l1_p
			train_loss += L.item()
			train_sim += sim.item()
			train_l1_z += l1_z.item()
			train_l1_p += l1_p.item()
			
			L.backward()
			self.optimizer.step()

		train_loss /= len(loader)
		train_sim /= len(loader)
		train_l1_z /= len(loader)
		train_l1_p /= len(loader)


		self.writer.add_scalar('train/total',train_loss,self.epoch)
		self.writer.add_scalar('train/sim',train_sim,self.epoch)
		self.writer.add_scalar('train/l1_z',train_l1_z,self.epoch)
		self.writer.add_scalar('train/l1_p',train_l1_p,self.epoch)

		
		print('Epoch {0:d} average train loss: {1:.3f}'.format(self.epoch,train_loss))
		print('Epoch {0:d} average train sim: {1:.3f}'.format(self.epoch,train_sim))
		print('Epoch {0:d} average train l1 (z): {1:.3f}'.format(self.epoch,train_l1_z))
		print('Epoch {0:d} average train l1 (p): {1:.3f}'.format(self.epoch,train_l1_p))

		return train_loss 

	def test_epoch(self,loader):

		self.eval()

		#loader.dataset.train_augment=True
		test_loss = 0.0
		test_sim = 0.0
		test_l1_z = 0.0
		test_l1_p = 0.0

		for ii,batch in enumerate(loader):

			(x1,x2) = batch
			
			x1,x2 = x1.unsqueeze(1).to(self.device),x2.unsqueeze(1).to(self.device)

			with torch.no_grad():
				z1,p1 = self.encode(x1)
				z2,p2 = self.encode(x2)

				sim1,l1_z1,l1_p1 = self.compute_loss(z1,p2)
				sim2,l1_z2,l1_p2 = self.compute_loss(z2,p1)
				sim = (sim1 + sim2)/2
				l1_z = (l1_z1 + l1_z2)/2
				l1_p = (l1_p1 + l1_p2)/2

				L = sim +  l1_p
				test_loss += L.item()
				test_sim += sim.item()
				test_l1_z += l1_z.item()
				test_l1_p += l1_p.item()

				test_loss += L.item()

		test_loss /= len(loader)
		test_sim /= len(loader)
		test_l1_z /= len(loader)
		test_l1_p /= len(loader)

		self.writer.add_scalar('test/total',test_loss,self.epoch)
		self.writer.add_scalar('test/sim',test_sim,self.epoch)
		self.writer.add_scalar('test/l1_z',test_l1_z,self.epoch)
		self.writer.add_scalar('test/l1_p',test_l1_p,self.epoch)

		print('Epoch {0:d} average test loss: {1:.3f}'.format(self.epoch,test_loss))
		print('Epoch {0:d} average test sim: {1:.3f}'.format(self.epoch,test_sim))
		print('Epoch {0:d} average test l1 (z): {1:.3f}'.format(self.epoch,test_l1_z))
		print('Epoch {0:d} average test l1 (p): {1:.3f}'.format(self.epoch,test_l1_p))

		return test_loss 


class masked_simsiam(simsiam):

	def __init__(self,encoder=None,predictor=None,sim_func=None,save_dir='',lr=1e-4,lamb=1e-10):

		"""
		simsiam for birdsong VAEs
		"""

		super(simsiam_l1,self).__init__(encoder=encoder,predictor=predictor,sim_func=sim_func,save_dir=save_dir,lr=lr)
		self.m = nn.Parameter(torch.empty(1,encoder.z_dim))
		nn.init.xavier_normal_(self.m)
		self.lamb = lamb
		self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir,'runs'))
		self.m.to(self.device)

	def compute_loss(self,z,p):

		z_loss = z.detach()
		l = self.sim_func(p,z_loss)

		tiled_m = torch.tile(self.m,(z.shape[0],1))
		l1_term_z = torch.abs(tiled_m * z_loss).sum(axis=1).mean()
		l1_term_p = torch.abs(p).sum(axis=1).mean()

		return l, l1_term_z, l1_term_p


	def train_epoch(self, loader):

		self.train()
		train_loss = 0.0
		train_sim = 0.0
		train_l1_z = 0.0
		train_l1_p = 0.0
		#loader.dataset.train_augment=True
		for ii, batch in enumerate(loader):

			(x1,x2) = batch
			
			x1,x2 = x1.unsqueeze(1).to(self.device),x2.unsqueeze(1).to(self.device)

			
			z1,p1 = self.encode(x1)
			z2,p2 = self.encode(x2)

			sim1,l1_z1,l1_p1 = self.compute_loss(z1,p2)
			sim2,l1_z2,l1_p2 = self.compute_loss(z2,p1)
			sim = (sim1 + sim2)/2
			l1_z = (l1_z1 + l1_z2)/2
			l1_p = (l1_p1 + l1_p2)/2

			L = sim + self.lamb * l1_z
			train_loss += L.item()
			train_sim += sim.item()
			train_l1_z += l1_z.item()
			train_l1_p += l1_p.item()
			
			L.backward()
			self.optimizer.step()

		train_loss /= len(loader)
		train_sim /= len(loader)
		train_l1_z /= len(loader)
		train_l1_p /= len(loader)


		self.writer.add_scalar('train/total',train_loss,self.epoch)
		self.writer.add_scalar('train/sim',train_sim,self.epoch)
		self.writer.add_scalar('train/l1_z',train_l1_z,self.epoch)
		self.writer.add_scalar('train/l1_p',train_l1_p,self.epoch)

		
		print('Epoch {0:d} average train loss: {1:.3f}'.format(self.epoch,train_loss))
		print('Epoch {0:d} average train sim: {1:.3f}'.format(self.epoch,train_sim))
		print('Epoch {0:d} average train l1 (z): {1:.3f}'.format(self.epoch,train_l1_z))
		print('Epoch {0:d} average train l1 (p): {1:.3f}'.format(self.epoch,train_l1_p))

		return train_loss 

	def test_epoch(self,loader):

		self.eval()

		#loader.dataset.train_augment=True
		test_loss = 0.0
		test_sim = 0.0
		test_l1_z = 0.0
		test_l1_p = 0.0

		for ii,batch in enumerate(loader):

			(x1,x2) = batch
			
			x1,x2 = x1.unsqueeze(1).to(self.device),x2.unsqueeze(1).to(self.device)

			with torch.no_grad():
				z1,p1 = self.encode(x1)
				z2,p2 = self.encode(x2)

				sim1,l1_z1,l1_p1 = self.compute_loss(z1,p2)
				sim2,l1_z2,l1_p2 = self.compute_loss(z2,p1)
				sim = (sim1 + sim2)/2
				l1_z = (l1_z1 + l1_z2)/2
				l1_p = (l1_p1 + l1_p2)/2

				L = sim + self.lamb*l1_z
				test_loss += L.item()
				test_sim += sim.item()
				test_l1_z += l1_z.item()
				test_l1_p += l1_p.item()

				test_loss += L.item()

		test_loss /= len(loader)
		test_sim /= len(loader)
		test_l1_z /= len(loader)
		test_l1_p /= len(loader)

		self.writer.add_scalar('test/total',test_loss,self.epoch)
		self.writer.add_scalar('test/sim',test_sim,self.epoch)
		self.writer.add_scalar('test/l1_z',test_l1_z,self.epoch)
		self.writer.add_scalar('test/l1_p',test_l1_p,self.epoch)

		print('Epoch {0:d} average test loss: {1:.3f}'.format(self.epoch,test_loss))
		print('Epoch {0:d} average test sim: {1:.3f}'.format(self.epoch,test_sim))
		print('Epoch {0:d} average test l1 (z): {1:.3f}'.format(self.epoch,test_l1_z))
		print('Epoch {0:d} average test l1 (p): {1:.3f}'.format(self.epoch,test_l1_p))

		return test_loss 	

class simsiam_l2(simsiam_l1):

	def __init__(self,encoder=None,predictor=None,sim_func=None,save_dir='',lr=1e-4,lamb=1e-10):

		"""
		simsiam for birdsong VAEs
		"""

		super(simsiam_l1,self).__init__(encoder=encoder,predictor=predictor,sim_func=sim_func,save_dir=save_dir,lr=lr)
		self.lamb = lamb
		self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir,'runs'))

	def compute_loss(self,z,p):

		z_loss = z.detach()
		l = self.sim_func(p,z_loss)

		l2_term_z = ((torch.pow(z_loss,2).sum(axis=1).sqrt() - 1)**2).mean()
		l2_term_p = ((torch.pow(p,2).sum(axis=1).sqrt() - 1)**2).mean()

		return l, l2_term_z, l2_term_p

class simsiam_normal_prior(simsiam):

        def __init__(self,encoder=None,predictor=None,sim_func=None,save_dir='',lr=1e-4):


                """
                simsiam with a standard normal prior
                """
                self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir,'runs'))

        def compute_loss(self,z,p):

                """
                the idea here is that simsiam with a standard normal prior will push things to have roughly zero mean, effectively normalizing our model

                TO DO
                Work out the math lm a o
                """
