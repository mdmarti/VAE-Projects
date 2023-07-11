import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

class latentSDE(nn.Module):
	
	def __init__(self,dim,diag_covar=True,device='cuda',save_dir=''):
		super(latentSDE,self).__init__()
		self.dim,self.diag_covar,self.device = dim,diag_covar,device
		self.save_dir = save_dir
		self.writer = SummaryWriter(log_dir = os.path.join(self.save_dir,'runs'))
		self.epoch=0

	def getEdX(self,evalPoint,dt):
		raise NotImplementedError

	def generate(self,data,T,dt):
		raise NotImplementedError 

	def loss(self,zt1,zt2,dt):
		raise NotImplementedError

	def forward(self,data):
		raise NotImplementedError
	
	def train_epoch(self,loader,optimizer):
		raise NotImplementedError
	
	def test_epoch(self,loader):
		raise NotImplementedError

	def save(self):
		raise NotImplementedError

	def load(self,fn):
		raise NotImplementedError 

class linearLatentSDE(latentSDE,nn.Module):

	def __init__(self,dim,diag_covar=True,save_dir=''):
		super(linearLatentSDE,self).__init__(dim,diag_covar,save_dir=save_dir)

		self.F = nn.Linear(self.dim,self.dim,bias=False)#torch.randn((self.dim,self.dim),requires_grad=True)
		self.D = torch.randn(self.dim) * torch.eye(self.dim)
		self.D.requires_grad = True

		#self.to(self.device)

	def getEdX(self, evalPoint, dt):
		return self.F(evalPoint)*dt
	
	def generate(self,z0,T,dt):
		t = np.arange(0,T,dt)
		zz = [z0]
		sample_dW = dt * torch.randn(len(t),self.dim)

		for jj in tqdm(range(len(t)),desc='generating sample trajectory'):
			
			with torch.no_grad():
				prev = zz[jj]
				
				prev = torch.from_numpy(prev).type(torch.FloatTensor)
				dz = self.F(prev)*dt + self.D @ sample_dW[jj,:] * prev

				zz.append(zz[jj] +dz.detach().cpu().numpy())

		zz = np.vstack(zz)
		return zz

	def loss(self,zt1,zt2,dt):
		

		mu = zt1 + self.F(zt1) * dt
		cov = self.D @ self.D.T * dt
		log_pzt2 = -1/2 * (((mu -zt2) @\
		      torch.diag(1/torch.diagonal(self.D * np.sqrt(dt))))**2)
		log_pzt2 = log_pzt2.sum(axis=-1)
		scale = -self.dim/2 * np.log(2*torch.pi) - 1/2*torch.log(torch.diagonal(cov)*dt).sum() 
		loss = - log_pzt2 - scale 
		
		return loss.sum() 
	
	def forward(self,data):

		zt1,zt2,dt = data 
		#assert(zt1.shape[0] == 128)

		loss = self.loss(zt1,zt2,dt)

		return loss
	
	def train_epoch(self, loader, optimizer):

		self.train()

		epoch_loss = 0.
		for batch in loader:
			loss = self.forward(batch)
			loss.backward()
			epoch_loss += loss.item()
			optimizer.step()

		self.writer.add_scalar('Train/loss',epoch_loss/len(loader),self.epoch)
		self.epoch += 1
	
		return epoch_loss,optimizer

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
	
class nonlinearLatentSDE(latentSDE,nn.Module):

	def __init__(self,dim,diag_covar=True,save_dir=''):
		super(nonlinearLatentSDE,self).__init__(dim,diag_covar,save_dir=save_dir)

		self.MLP = nn.Sequential(nn.Linear(self.dim,100),
			   					nn.Softplus(),
								nn.Linear(100,self.dim))
		self.D = nn.Sequential(nn.Linear(self.dim,100),
			 					nn.Softplus(),
								nn.Linear(100,self.dim),
								nn.Softplus())
		

		#self.to(self.device)

	def generate(self,z0,T,dt):
		t = np.arange(0,T,dt)
		zz = [z0]
		sample_dW =  torch.randn(len(t),self.dim)

		for jj in range(len(t)):
			
			with torch.no_grad():
				prev = zz[jj]
				
				prev = torch.from_numpy(prev).type(torch.FloatTensor)
				D = self.D(prev) *np.sqrt(dt)
				dz = self.MLP(prev)*dt + D @ sample_dW[jj,:]

				zz.append(zz[jj] +dz.detach().cpu().numpy())

		zz = np.vstack(zz)
		return zz

	def loss(self,zt1,zt2,dt):
		
		dt = dt[0]
		dzTrue = zt2 - zt1
		mu = self.MLP(zt1) * dt
		D = self.D(zt1) *np.sqrt(dt)
		cov = D * D
		log_pzt2 = -1/2 * (((mu-dzTrue) *(1/D))**2)
		log_pzt2 = log_pzt2.sum(axis=-1)
		scale = -self.dim/2 * np.log(2*torch.pi)\
			  - 1/2*torch.log(cov).sum() 
		loss = - log_pzt2 - scale 
		
		return loss.sum() 
	
	def forward(self,data):

		zt1,zt2,dt = data 
		#assert(zt1.shape[0] == 128)

		loss = self.loss(zt1,zt2,dt)

		return loss
	
	def train_epoch(self, loader, optimizer):

		self.train()

		epoch_loss = 0.
		for batch in loader:
			loss = self.forward(batch)
			loss.backward()
			epoch_loss += loss.item()
			optimizer.step()

		self.writer.add_scalar('Train/loss',epoch_loss/len(loader),self.epoch)
		self.epoch += 1
	
		return epoch_loss,optimizer

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
		check = torch.load(fn)#,map_location=self.device)
		self.epoch = check['epoch']
		self.load_state_dict(check['model_state_dict'])
		return


