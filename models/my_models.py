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
		self.n_entries = np.sum(list(range(1,self.dim+1)))
		if dim ==1: assert self.n_entries == 1
		elif dim==2: assert self.n_entries ==3
		elif dim == 3: assert self.n_entries==6
		elif dim == 4: assert self.n_entries == 10
		self.tril_inds = torch.tril_indices(dim,dim)
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
		self.D = nn.Linear(self.dim,self.n_entries,bias=False)

		self.to(self.device)

	def getEdX(self, evalPoint, dt):
		return self.F(evalPoint)*dt
	
	def generate(self,z0,T,dt):
		t = np.arange(0,T,dt)
		zz = [z0]
		sample_dW = dt * torch.randn(len(t),self.dim).to(self.device)

		for jj in tqdm(range(len(t)),desc='generating sample trajectory'):
			
			with torch.no_grad():
				prev = zz[jj]
				
				prev = torch.from_numpy(prev).type(torch.FloatTensor)
				dz = self.F(prev)*dt + self.D @ sample_dW[jj,:] * prev

				zz.append(zz[jj] +dz.detach().cpu().numpy())

		zz = np.vstack(zz)
		return zz

	def loss(self,zt1,zt2,dt):
		
		### Loss target ####
		dzTrue = (zt2-zt1).view(zt1.shape[0],zt1.shape[1],1)
		### estimate mu ####
		mu = self.F(zt1) * dt
		mu=mu.view(mu.shape[0],mu.shape[1],1)
		### estimate cholesky factor ####
		chol = torch.zeros(zt1.shape[0],self.dim,self.dim).to(self.device)
		D = torch.exp(self.D(zt1))
		chol[:,self.tril_inds[0],self.tril_inds[1]] = D*np.sqrt(dt)
		####### invert cholesky factor #######
		eyes = torch.eye(self.dim).view(1,self.dim,self.dim).repeat(zt1.shape,1,1).to(self.device)
		invChol = torch.linalg.solve_triangular(chol,eyes)
		###### get precision and covariance #####
		precision = invChol.transpose(-2,-1) @ invChol
		cov = chol.transpose(-2,1) @ chol
		##### calculate loss ###################
		const = -self.dim/2 * np.log(2*torch.pi)
		t1 = torch.logdet(cov).squeeze()
		assert len(t1.shape) == 1, print(t1.shape)
		t2 = (dzTrue.transpose(-2,-1) @ precision @ dzTrue).squeeze()
		assert len(t2.shape)==1,print(t2.shape)
		t3 = -2*(dzTrue.transpose(-2,-1) @ precision @ mu).squeeze()
		assert len(t3.shape) == 1,print(t3.shape)
		t4 = (mu.transpose(-2,-1)@precision @ mu).squeeze()
		assert len(t4.shape) == 1,print(t4.shape)
		log_pz2 = const - 1/2*(t1 + t2 + t3 + t4)
		loss = - log_pz2
		
		
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

	def __init__(self,dim,diag_covar=True,save_dir='',true1=[1.],true2=[0.5],p1name='mu',p2name='sigma'):
		
		super(nonlinearLatentSDE,self).__init__(dim,diag_covar,save_dir=save_dir)


		self.MLP = nn.Linear(self.dim,self.dim)
		#nn.Sequential(nn.Linear(self.dim,100),
		#	   					nn.Softplus(),
		#						nn.Linear(100,self.dim))
		self.D = nn.Linear(self.dim,self.n_entries)
		#nn.Sequential(nn.Linear(self.dim,100),
		#	 					nn.Softplus(),
		#						nn.Linear(100,self.n_entries))
		

		self.p1name = p1name
		self.p2name = p2name
		self.true1=true1
		self.true2 = true2
		"""
		layout = {
			'Train': {
				"loss":['Multiline',['logp']]
			},
			'Test': {
				"loss":['Multiline',['log p']]
			}
		}
		for d in range(self.dim):
			layout['Train'][f"{self.p1name} dim {d+1}"] = \
				["Multiline",[f"{self.p1name} dim {d+1}/estimated", f"{self.p1name} dim {d+1}/true"]]
			layout['Test'][f"{self.p1name} dim {d+1}"] = \
				["Multiline",[f"{self.p1name} dim {d+1}/estimated", f"{self.p1name} dim {d+1}/true"]]
		self.writer.add_custom_scalars(layout)
		"""
		self.to(self.device)

	def generate(self,z0,T,dt):
		t = np.arange(0,T,dt)
		zz = [z0]
		sample_dW =  np.sqrt(dt) * torch.randn(len(t),self.dim).to(self.device)

		for jj in range(len(t)):
			
			with torch.no_grad():
				prev = zz[jj]
				
				prev = torch.from_numpy(prev).type(torch.FloatTensor).to(self.device)
				chol = torch.zeros(prev.shape[0],self.dim,self.dim).to(self.device)
				D = torch.exp(self.D(prev))
				chol[:,self.tril_inds[0],self.tril_inds[1]] = D 
				dz = self.MLP(prev)*dt + chol @ sample_dW[jj,:]

				zz.append(zz[jj] +dz.detach().cpu().numpy())

		zz = np.vstack(zz)
		return zz

	def loss(self,zt1,zt2,dt):
		
		"""
		z_t's: batch x z_dim
		mu: batch x z_dim x 1
		D: batch x z_dim x 1
		
		"""

		
		dt = dt[0]

		### Loss target ####
		dzTrue = (zt2-zt1).view(zt1.shape[0],zt1.shape[1],1)
		### estimate mu ####
		mu = (self.MLP(zt1) * dt)
		mu = mu.view(mu.shape[0],mu.shape[1],1)
		### estimate cholesky factor ####
		L = torch.zeros(zt1.shape[0],self.dim,self.dim).to(self.device)
		D = torch.exp(self.D(zt1))
		L[:,self.tril_inds[0],self.tril_inds[1]] = D 
		####### invert cholesky factor #######
		eyes = torch.eye(self.dim).view(1,self.dim,self.dim).repeat(zt1.shape[0],1,1).to(self.device)
		invChol = torch.linalg.solve_triangular(L,eyes,upper=False)
		###### get precision and covariance #####
		precision = invChol.transpose(-2,-1) @ invChol / dt
		cov = L @ L.transpose(-2,-1) * dt 
		##### calculate loss ###################

		const = -self.dim/2 * np.log(2*torch.pi)
		t1 = torch.logdet(cov).squeeze()
		assert len(t1.shape) == 1, print(t1.shape)
		t2 = (dzTrue.transpose(-2,-1) @ precision @ dzTrue).squeeze()
		assert len(t2.shape)==1,print(t2.shape)
		t3 = -2*(dzTrue.transpose(-2,-1) @ precision @ mu).squeeze()
		assert len(t3.shape) == 1,print(t3.shape)
		t4 = (mu.transpose(-2,-1)@precision @ mu).squeeze()
		assert len(t4.shape) == 1,print(t4.shape)
		log_pz2 = const - 1/2*(t1 + t2 + t3 + t4)
		loss = - log_pz2

		###########################################

		return loss.sum(),mu.view(len(zt1),self.dim)/(zt1.view(len(zt1),self.dim)*dt),\
			torch.diagonal(L,dim1=-2,dim2=-1).view(len(zt1),self.dim)/(zt1.view(len(zt1),self.dim))
	
	def forward(self,data):

		zt1,zt2,dt = data
		zt1,zt2,dt = zt1.to(self.device),zt2.to(self.device),dt.to(self.device) 
		#assert(zt1.shape[0] == 128)

		loss,F,D = self.loss(zt1,zt2,dt)


		return loss,F,D
	
	def train_epoch(self, loader, optimizer):

		self.train()

		epoch_loss = 0.
		epoch_mus = []
		epoch_sigs = []
		for batch in loader:
			loss,mu,sig = self.forward(batch)
			loss.backward()
			epoch_loss += loss.item()
			optimizer.step()
			epoch_mus.append(mu.detach().cpu().numpy())
			epoch_sigs.append(sig.detach().cpu().numpy())

		epoch_mus = np.nanmean(np.vstack(epoch_mus),axis=0)
		epoch_sigs = np.nanmean(np.vstack(epoch_sigs),axis=0)

		self.writer.add_scalar('Train/loss',epoch_loss/len(loader),self.epoch)
		for d in range(self.dim):
			self.writer.add_scalars(f'Train/{self.p1name} dim {d+1}',{'estimated':epoch_mus[d],'true':self.true1[d]},self.epoch)
			self.writer.add_scalars(f'Train/{self.p2name} dim {d+1}',{'estimated':epoch_sigs[d],'true':self.true2[d]},self.epoch)
		self.epoch += 1
	
		return epoch_loss,optimizer
	
	def test_epoch(self,loader):

		self.eval()
		with torch.no_grad():
			epoch_loss = 0.
			epoch_mus = []
			epoch_sigs = []
			for batch in loader:
				loss,mu,sig = self.forward(batch)
				
				epoch_loss += loss.item()

				epoch_mus.append(mu.detach().cpu().numpy())
				epoch_sigs.append(sig.detach().cpu().numpy())

		epoch_mus = np.nanmean(np.vstack(epoch_mus),axis=0)
		epoch_sigs = np.nanmean(np.vstack(epoch_sigs),axis=0)

		self.writer.add_scalar('Test/loss',epoch_loss/len(loader),self.epoch)
		for d in range(self.dim):
			self.writer.add_scalars(f'Test/{self.p1name} dim {d+1}',{'estimated':epoch_mus[d],'true':self.true1[d]},self.epoch)
			self.writer.add_scalars(f'Test/{self.p2name} dim {d+1}',{'estimated':epoch_sigs[d],'true':self.true2[d]},self.epoch)
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

class Simple1dTestDE(nonlinearLatentSDE,nn.Module):

	def __init__(self,dim=1,diag_covar=True,save_dir='test'):
		super(Simple1dTestDE,self).__init__(dim,diag_covar,save_dir=save_dir)

		self.p1name = 'mu'
		self.p2name = 'sigma'

	def generate(self, z0, T, dt):
		t = np.arange(0,T,dt)
		zz = [z0]
		sample_dW =  np.sqrt(dt) * torch.randn(len(t),self.dim).to(self.device)

		for jj in range(len(t)):
			
			with torch.no_grad():
				prev = zz[jj]
				prev = torch.from_numpy(prev).type(torch.FloatTensor).to(self.device)

				mu = self.MLP(prev) * dt
				sigma = torch.exp(self.D(prev))

				dz = mu + sigma *  sample_dW[jj,:]

				zz.append(zz[jj] +dz.detach().cpu().numpy())

		zz = np.vstack(zz)
		return zz
	
	def loss(self, zt1, zt2, dt):
		"""
		instead of parameterizing mu, Sigma, parameterize
		Lambda, eta (lambda * mu)
		"""

		dt = dt[0]
		### Loss target ####
		dzTrue = (zt2-zt1)#.view(zt1.shape[0],zt1.shape[1],1)
		### estimate mu ####
		mu = self.MLP(zt1) * dt
		#mu = mu.view(mu.shape[0],mu.shape[1],1)
		### estimate cholesky factor ####
		sigma = torch.exp(self.D(zt1)) * torch.sqrt(dt)
		#### Covariance ###########
		cov = sigma**2
		###### Precision #####
		precision = 1/cov
		
		##### calculate loss ###################
		c = -self.dim/2 * np.log(2*torch.pi)
		t1 = torch.log(cov).squeeze()
		assert (len(t1.shape) == 1) & (len(t1) == len(zt1)), print(t1.shape)
		t2 = ((dzTrue **2) * precision).squeeze()
		assert (len(t2.shape)==1) &(len(t2) == len(zt1)),print(t2.shape)
		t3 = -2*(dzTrue * precision * mu).squeeze()
		assert (len(t3.shape) == 1) &(len(t3) == len(zt1)),print(t3.shape)
		t4 = (mu**2 * precision).squeeze()
		assert (len(t4.shape) == 1) &(len(t4) == len(zt1)),print(t4.shape)
		log_pz2 = c - 1/2*(t1 + t2 + t3+t4)
		loss = - log_pz2
		###########################################
		return loss.sum(),mu/(zt1*dt),sigma/(zt1 * torch.sqrt(dt))
			



class nonlinearLatentSDENatParams(nonlinearLatentSDE,nn.Module):

	def __init__(self,dim,diag_covar=True,save_dir=''):
		
		super(nonlinearLatentSDENatParams,self).__init__(dim,diag_covar,save_dir=save_dir)

		self.p1name = 'eta'
		self.p2name = 'lambda'
	def generate(self, z0, T, dt):
		t = np.arange(0,T,dt)
		zz = [z0]
		sample_dW =  np.sqrt(dt) * torch.randn(len(t),self.dim).to(self.device)

		for jj in range(len(t)):
			
			with torch.no_grad():
				prev = zz[jj]
				prev = torch.from_numpy(prev).type(torch.FloatTensor).to(self.device)

				eta = self.MLP(prev)
				D = torch.exp(self.D(prev))
				L = torch.zeros(self.dim,self.dim).to(self.device)
				L[self.tril_inds[0],self.tril_inds[1]] = D 
				eyes = torch.eye(self.dim).to(self.device)
				LInv = torch.linalg.solve_triangular(L,eyes,upper=False)
				
				cov = LInv.transpose(-2,-1) @ LInv
				mu = cov @ eta *dt
				dz = mu + LInv.transpose(-2,-1) @ sample_dW[jj,:]

				zz.append(zz[jj] +dz.detach().cpu().numpy())

		zz = np.vstack(zz)
		return zz
	
	def loss(self, zt1, zt2, dt):
		"""
		instead of parameterizing mu, Sigma, parameterize
		Lambda, eta (lambda * mu)
		"""

		dt = dt[0]
		### Loss target ####
		dzTrue = (zt2-zt1).view(zt1.shape[0],zt1.shape[1],1)
		### estimate eta ####
		eta = self.MLP(zt1)
		eta = eta.view(eta.shape[0],eta.shape[1],1)
		### estimate cholesky factor ####
		D = torch.exp(self.D(zt1))
		L = torch.zeros(zt1.shape[0],self.dim,self.dim).to(self.device)
		L[:,self.tril_inds[0],self.tril_inds[1]] = D 
		#### Precision ###########
		precision = L @ L.transpose(-2,-1) /dt
		####### invert cholesky factor #######
		eyes = torch.eye(self.dim).view(1,self.dim,self.dim).repeat(zt1.shape[0],1,1).to(self.device)
		LInv = torch.linalg.solve_triangular(L,eyes,upper=False)
		###### covariance #####
		cov = LInv.transpose(-2,-1) @ LInv * dt
		
		##### calculate loss ###################
		c = -self.dim/2 * np.log(2*torch.pi)
		t1 = -torch.logdet(precision).squeeze()
		assert (len(t1.shape) == 1) & (len(t1) == len(zt1)), print(t1.shape)
		t2 = (dzTrue.transpose(-2,-1) @ precision @ dzTrue).squeeze()
		assert (len(t2.shape)==1) &(len(t2) == len(zt1)),print(t2.shape)
		t3 = -2*(dzTrue.transpose(-2,-1) @ eta).squeeze()
		assert (len(t3.shape) == 1) &(len(t3) == len(zt1)),print(t3.shape)
		t4 = (eta.transpose(-2,-1)@cov @ eta).squeeze()
		assert (len(t4.shape) == 1) &(len(t4) == len(zt1)),print(t4.shape)
		log_pz2 = c - 1/2*(t1 + t2 + t3+t4)
		loss = - log_pz2
		###########################################
		return loss.sum(),eta.view(len(zt1),self.dim),\
			torch.diagonal(L,dim1=-2,dim2=-1).view(len(zt1),self.dim)*torch.sqrt(dt)/zt1.view(len(zt1),self.dim)


