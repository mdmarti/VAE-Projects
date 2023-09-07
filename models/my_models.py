import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

def diag_indices(dim):

	return torch.eye(dim,dtype=torch.uint8).nonzero().transpose(-2,-1)

class latentSDE(nn.Module):
	
	def __init__(self,dim,device='cuda',save_dir='',diag=True):
		super(latentSDE,self).__init__()
		self.diag = diag
		self.dim,self.device = dim,device

		if self.diag:
			self.chol_inds = diag_indices(dim)
			self.n_entries=self.dim 
		else:
			self.chol_inds = torch.tril_indices(dim,dim)
			self.n_entries = np.sum(list(range(1,self.dim+1)))
			if dim ==1: assert self.n_entries == 1
			elif dim==2: assert self.n_entries ==3
			elif dim == 3: assert self.n_entries==6
			elif dim == 4: assert self.n_entries == 10
		self.save_dir = save_dir
		self.writer = SummaryWriter(log_dir = os.path.join(self.save_dir,'runs'))
		self.epoch=0

	def getMoments(self,data):
		"""
		get estimates of moments given data
		"""
		raise NotImplementedError
	
	def getNatParams(self,data):
		"""
		get estimates of natural parameters given data
		"""
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

	def __init__(self,dim,save_dir='',diag=True):
		super(linearLatentSDE,self).__init__(dim,save_dir=save_dir,diag=diag)

		self.MLP = nn.Linear(self.dim,self.dim,bias=False)#torch.randn((self.dim,self.dim),requires_grad=True)
		self.D = nn.Linear(self.dim,self.n_entries,bias=False)

		self.to(self.device)

	def getMoments(self, data):
		"""
		get estimates of moments given data
		input: data
		output: mu,L (lower triangular factor of covariance)
		"""
		mu = self.MLP(data)
		### estimate cholesky factor ####
		chol = torch.zeros(data.shape[0],self.dim,self.dim).to(self.device)
		D = torch.exp(self.D(data))
		chol[:,self.chol_inds[0],self.chol_inds[1]] = D
		return mu,chol
	
	def getNatParams(self, data):
		mu = self.MLP(data).view(data.shape[0],data.shape[1],1)
		### estimate cholesky factor ####
		chol = torch.zeros(data.shape[0],self.dim,self.dim).to(self.device)
		D = torch.exp(self.D(data))
		chol[:,self.chol_inds[0],self.chol_inds[1]] = D
		####### invert cholesky factor #######
		eyes = torch.eye(self.dim).view(1,self.dim,self.dim).repeat(data.shape,1,1).to(self.device)
		invChol = torch.linalg.solve_triangular(chol,eyes)
		###### get precision and covariance #####
		precision = invChol.transpose(-2,-1) @ invChol
		return (precision @ mu).squeeze(),invChol
	
	def generate(self,z0,T,dt):
		t = np.arange(0,T,dt)
		zz = [z0]
		sample_dW = np.sqrt(dt) * torch.randn(len(t),self.dim).to(self.device)

		for jj in tqdm(range(len(t)),desc='generating sample trajectory'):
			
			with torch.no_grad():
				prev = zz[jj]
				
				prev = torch.from_numpy(prev).type(torch.FloatTensor)
				### estimate cholesky factor ####
				chol = torch.zeros(prev.shape[0],self.dim,self.dim).to(self.device)
				D = torch.exp(self.D(prev))
				chol[:,self.chol_inds[0],self.chol_inds[1]] = D
				dz = self.MLP(prev)*dt + chol @ sample_dW[jj,:]

				zz.append(zz[jj] +dz.detach().cpu().numpy())

		zz = np.vstack(zz)
		return zz

	def loss(self,zt1,zt2,dt):
		
		### Loss target ####
		dzTrue = (zt2-zt1).view(zt1.shape[0],zt1.shape[1],1)
		### estimate mu ####
		mu = self.MLP(zt1) * dt
		mu=mu.view(mu.shape[0],mu.shape[1],1)
		### estimate cholesky factor ####
		chol = torch.zeros(zt1.shape[0],self.dim,self.dim).to(self.device)
		D = torch.exp(self.D(zt1))
		chol[:,self.chol_inds[0],self.chol_inds[1]] = D*np.sqrt(dt)
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

	def __init__(self,dim:int,save_dir:str ='',true1:"function"=None,true2:"function"=None,
	      p1name:str='mu',p2name:str='sigma',plotDists:bool=True,diag:bool=True):
		
		super(nonlinearLatentSDE,self).__init__(dim,save_dir=save_dir,diag=diag)


		self.MLP = nn.Sequential(nn.Linear(self.dim,100),
			   					nn.Softplus(),
								nn.Linear(100,self.dim))
		self.D = nn.Sequential(nn.Linear(self.dim,100),
			 					nn.Softplus(),
								nn.Linear(100,self.n_entries))
		

		self.p1name = p1name
		self.p2name = p2name
		self.true1=true1
		self.true2 = true2
		self.plotDists=plotDists

		self.to(self.device)

	def _add_dist_figure(self,estimates:np.ndarray,ground:np.ndarray,name:str,dim:int,epoch_type:str='Train'):
		plt.rcParams.update({'font.size': 22})
		#estimates = estimates.detach().cpu().numpy()
		#ground = ground.detach().cpu().numpy()
		assert len(ground) > 1
		assert len(estimates) > 1
		fig1 = plt.figure(figsize=(10,10))

		plt.figure(fig1)
		ax = plt.gca()
		sns.kdeplot(data=ground,color="#58445F",alpha=0.8,label=f"Ground truth {name}",ax=ax,warn_singular=False)
		sns.kdeplot(data=estimates,color="#3B93B3",alpha=0.8,label=f"Model estimated {name}",ax=ax)
		ax.set_xlabel(f"{name}")
		ax.set_ylabel("Density")
		ax.set_yticks([])
		rangeVals = np.amax(estimates) - np.amin(estimates)
		ax.set_xlim([np.amin(estimates) - 3*rangeVals,np.amax(estimates) + 3*rangeVals])
		plt.legend()

		self.writer.add_figure(f'{epoch_type}/{name} dim {dim}',fig1,close=True,global_step=self.epoch)
		return


	def getMoments(self, data: torch.FloatTensor):
		"""
		get estimates of moments given data
		input: data
		output: mu,L (lower triangular factor of covariance)
		"""
		mu = self.MLP(data)
		### estimate cholesky factor ####
		chol = torch.zeros(data.shape[0],self.dim,self.dim).to(self.device)
		D = torch.exp(self.D(data))
		chol[:,self.chol_inds[0],self.chol_inds[1]] = D
		return mu,chol
	
	def getNatParams(self, data: torch.FloatTensor):

		"""
		estimate natural parameters given data
		"""
		mu = self.MLP(data).view(data.shape[0],data.shape[1],1)
		### estimate cholesky factor ####
		chol = torch.zeros(data.shape[0],self.dim,self.dim).to(self.device)
		D = torch.exp(self.D(data))
		chol[:,self.chol_inds[0],self.chol_inds[1]] = D
		####### invert cholesky factor #######
		eyes = torch.eye(self.dim).view(1,self.dim,self.dim).repeat(data.shape,1,1).to(self.device)
		invChol = torch.linalg.solve_triangular(chol,eyes)
		###### get precision and covariance #####
		precision = invChol.transpose(-2,-1) @ invChol
		return (precision @ mu).squeeze(),invChol
	

	def generate(self,z0,T,dt):
		t = np.arange(0,T,dt)
		zz = [z0]
		

		for jj in range(1,len(t)+1):
			
			with torch.no_grad():
				prev = zz[jj-1]
				prev = torch.from_numpy(prev).type(torch.FloatTensor).to(self.device).view(1,len(prev))
				#print(prev.shape)
				mu,L = self.getMoments(prev)
				mu = mu.view(1,mu.shape[1])
				sample_dW =  np.sqrt(dt) * torch.randn(self.dim).to(self.device)
				dz = mu*dt + L @ sample_dW
				#prev = torch.from_numpy(prev).type(torch.FloatTensor).to(self.device)
				#chol = torch.zeros(self.dim,self.dim).to(self.device)
				#D = self.D(prev)
				#chol[self.chol_inds[0],self.chol_inds[1]] = D 
				#dz = self.MLP(prev)*dt + chol @ sample_dW[jj,:]

				zz.append(zz[jj-1] +dz.detach().cpu().numpy().squeeze())

		zz = np.vstack(zz)
		return zz

	def loss(self,zt1,zt2,dt):
		
		"""
		z_t's: batch x z_dim
		mu: batch x z_dim x 1
		D: batch x z_dim x 1
		
		returns loss for the batch,
		mu for the batch, D for the batch
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
		L[:,self.chol_inds[0],self.chol_inds[1]] = D 
		####### invert cholesky factor #######
		eyes = torch.eye(self.dim).view(1,self.dim,self.dim).repeat(zt1.shape[0],1,1).to(self.device)
		invChol = torch.linalg.solve_triangular(L,eyes,upper=False)
		###### get precision and covariance #####
		precision = invChol.transpose(-2,-1) @ invChol 
		cov = L @ L.transpose(-2,-1) 
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

		return loss.sum(),mu.view(len(zt1),self.dim),D * torch.sqrt(dt)
	
	def forward(self,data: torch.FloatTensor):

		zt1,zt2,dt = data
		zt1,zt2,dt = zt1.to(self.device),zt2.to(self.device),dt.to(self.device) 

		loss,F,D = self.loss(zt1,zt2,dt)

		return loss,F,D
	
	def train_epoch(self, loader, optimizer):

		self.train()

		epoch_loss = 0.
		epoch_mus = []
		epoch_Ds = []
		true_p1 = []
		true_p2 = []
		for batch in loader:
			loss,mu,d = self.forward(batch)
			loss.backward()
			epoch_loss += loss.item()
			optimizer.step()
			epoch_mus.append(mu.detach().cpu().numpy())
			epoch_Ds.append(d.detach().cpu().numpy())
			
			true_p1.append(self.true1(batch[0]))
			true_p2.append(self.true2(batch[0]))

		epoch_mus = np.vstack(epoch_mus)
		epoch_Ds = np.vstack(epoch_Ds)
		true_p1 = np.vstack(true_p1)
		true_p2 = np.vstack(true_p2)

		self.writer.add_scalar('Train/loss',epoch_loss/len(loader),self.epoch)
		if self.plotDists & (self.epoch % 50 == 0):
			for d in range(self.dim):
				self._add_dist_figure(epoch_mus[:,d],true_p1[:,d],self.p1name,d+1,'Train')
				#self.writer.add_scalars(f'Train/{self.p2name} dim {d+1}',{'estimated':epoch_sigs[d],'true':self.true2[d]},self.epoch)
			for d in range(self.n_entries):
				self._add_dist_figure(epoch_Ds[:,d],true_p2[:,d],self.p2name,d+1,'Train')
		self.epoch += 1
	
		return epoch_loss,optimizer
	
	def test_epoch(self,loader):

		self.eval()
		with torch.no_grad():
			epoch_loss = 0.
			epoch_mus = []
			epoch_Ds = []
			true_p1 = []
			true_p2 = []
			for batch in loader:
				loss,mu,d = self.forward(batch)
				
				epoch_loss += loss.item()

				epoch_mus.append(mu.detach().cpu().numpy())
				epoch_Ds.append(d.detach().cpu().numpy())
				true_p1.append(self.true1(batch[0]))
				true_p2.append(self.true2(batch[0]))
		epoch_mus = np.vstack(epoch_mus)
		epoch_Ds = np.vstack(epoch_Ds)
		true_p1 = np.vstack(true_p1)
		true_p2 = np.vstack(true_p2)

		self.writer.add_scalar('Test/loss',epoch_loss/len(loader),self.epoch)
		if self.plotDists:
			for d in range(self.dim):
				self._add_dist_figure(epoch_mus[:,d],true_p1[:,d],self.p1name,d+1,'Test')
				#self.writer.add_scalars(f'Train/{self.p2name} dim {d+1}',{'estimated':epoch_sigs[d],'true':self.true2[d]},self.epoch)
			for d in range(self.n_entries):
				self._add_dist_figure(epoch_Ds[:,d],true_p2[:,d],self.p2name,d+1,'Test')
		"""
		for d in range(self.dim):
			if self.plotDists:
				self.writer.add_scalars(f'Test/{self.p1name} dim {d+1}',{'estimated':epoch_mus[d],'true':self.true1[d]},self.epoch)
				self.writer.add_scalars(f'Test/{self.p2name} dim {d+1}',{'estimated':epoch_sigs[d],'true':self.true2[d]},self.epoch)
			else:
				self.writer.add_scalar(f'Test/{self.p1name} dim {d+1}',epoch_mus[d],self.epoch)
				self.writer.add_scalar(f'Test/{self.p2name} dim {d+1}',epoch_sigs[d],self.epoch)
		"""
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

	def __init__(self,dim=1,save_dir='test'):
		super(Simple1dTestDE,self).__init__(dim,save_dir=save_dir)

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
		NORMAL LOSS
		
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

	def __init__(self,dim:int,save_dir:str ='',true1:"function"=None,true2:"function"=None,
	      p1name:str='mu',p2name:str='sigma',plotDists:bool=True,diag:bool=True):
		
		super(nonlinearLatentSDENatParams,self).__init__(dim,save_dir=save_dir,\
						   p1name=p1name,p2name=p2name,true1=true1,true2=true2,plotDists=plotDists,diag=diag)

	def getMoments(self, data):
		"""
		get estimates of moments given data
		input: data (batch size x dim)
		output: mu,L (lower triangular factor of covariance)
		"""
		eta = self.MLP(data).view(data.shape[0],data.shape[1],1)
		### estimate cholesky factor ####
		chol = torch.zeros(data.shape[0],self.dim,self.dim).to(self.device)
		D = torch.exp(self.D(data))
		chol[:,self.chol_inds[0],self.chol_inds[1]] = D
		eyes = torch.eye(self.dim).view(1,self.dim,self.dim).repeat(data.shape[0],1,1).to(self.device)
		invChol = torch.linalg.solve_triangular(chol,eyes,upper=False)
				
		cov = invChol.transpose(-2,-1) @ invChol

		return cov @ eta,invChol
	
	def getNatParams(self, data):
		eta = self.MLP(data)
		### estimate cholesky factor ####
		chol = torch.zeros(data.shape[0],self.dim,self.dim).to(self.device)
		D = torch.exp(self.D(data))
		chol[:,self.chol_inds[0],self.chol_inds[1]] = D
		
		return eta,chol

	def generate(self, z0, T, dt):
		t = np.arange(0,T,dt)
		zz = [z0]
		sample_dW =  np.sqrt(dt) * torch.randn(len(t),self.dim).to(self.device)

		for jj in range(len(t)):
			
			with torch.no_grad():
				prev = zz[jj]
				
				prev = torch.from_numpy(prev).type(torch.FloatTensor).to(self.device).view(1,len(prev))
				#print(prev.shape)
				mu,L = self.getMoments(prev)
				mu = mu.view(1,mu.shape[1])
				#print(L.shape)
				#print(sample_dW[jj,:].shape)
				
				"""
				chol = torch.zeros(self.dim,self.dim).to(self.device)
				
				D = self.D(prev)
				
				chol[self.chol_inds[0],self.chol_inds[1]] = D
 
				eyes = torch.eye(self.dim).to(self.device)
				invChol = torch.linalg.solve_triangular(chol,eyes,upper=False)
				
				eta = self.MLP(prev)
				
				cov = invChol.transpose(-2,-1) @ invChol
				
				mu = cov @ eta *dt
				"""
				#print(mu.shape)
				#print(L.shape)
				dz = mu*dt + L @ sample_dW[jj,:]
				#print(dz.shape)
				zz.append(zz[jj] +dz.detach().cpu().numpy().squeeze())

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
		L[:,self.chol_inds[0],self.chol_inds[1]] = D 
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
		return loss.sum(),eta.view(len(zt1),self.dim),D/torch.sqrt(dt)


