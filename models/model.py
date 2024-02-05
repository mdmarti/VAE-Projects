from latent_sde import *
from encoding import *
from tqdm import tqdm
from torch.func import jacrev,vmap

EPS = 1e-8
class EmbeddingSDE(nn.Module):


	def __init__(self,dataDim,latentDim,encoding,latentSDE,
				 device='cuda'):

		super(EmbeddingSDE,self).__init__()
		self.sde = latentSDE
		self.encoder =encoding 
		self.dataDim=dataDim
		self.latentDim=latentDim 
		self.device=device
		self.mu=10000
		self.k = 500
		self.gamma = self.latentDim
		self.covarGamma = 0.5
		self.to(self.device)
		self.counter = 0.
		print("using full entropy version")

	def _init_batch_covar_approx(self,batch_size=512,epsilon = 0.2):

		"""
		create our first estimate of the batch covariance!
		
		initializes as the Bsz x Bsz identity, to begin with
		"""

		self.batch_size= batch_size
		self.batch_approx = torch.eye(batch_size).to(self.device)
		self.epsilon=epsilon

	def _add_dist_figure(self,estimates:np.ndarray,name:str,dim:int,epoch_type:str='Train'):
		plt.rcParams.update({'font.size': 22})
		#estimates = estimates.detach().cpu().numpy()
		#ground = ground.detach().cpu().numpy()
		
		assert len(estimates) > 1
		
		fig1 = plt.figure(figsize=(10,10))

		plt.figure(fig1)
		ax = plt.gca()
		sns.kdeplot(data=estimates,color="#3B93B3",alpha=0.8,label=f"Model estimated {name}",ax=ax)
		ax.set_xlabel(f"{name}")
		ax.set_ylabel("Density")
		ax.set_yticks([])
		rangeVals = np.amax(estimates) - np.amin(estimates)
		#ax.set_xlim([np.amin(estimates) - 3*rangeVals,np.amax(estimates) + 3*rangeVals])
		plt.legend()

		self.sde.writer.add_figure(f'{epoch_type}/{name} dim {dim}',fig1,close=True,global_step=self.sde.epoch)
		

	def _add_quiver(self,data:np.ndarray,estimates:np.ndarray,name:str,epoch_type:str='Train'):
		
		fig2 = plt.figure(figsize=(10,10))

		plt.figure(fig2)
		ax = plt.gca()
		
		ax.quiver(data[:,0],data[:,1],estimates[:,0],estimates[:,1],color="#3B93B3",label=f"Model estimated {name}")
		ax.set_xlabel(f"{name} 1")
		ax.set_ylabel(f"{name} 2")
		#ax.set_yticks([])
		#ax.set_xticks([])
		#rangeVals = np.amax(estimates) - np.amin(estimates)
		#ax.set_xlim([np.amin(estimates) - 3*rangeVals,np.amax(estimates) + 3*rangeVals])
		plt.legend()
		
		self.sde.writer.add_figure(f'{epoch_type}/{name} quiver',fig2,close=True,global_step=self.sde.epoch)

		return
	
	def _reg_sd(self,data,mu,epsilon):

		n = data.shape[0]
		var = ((data - mu)**2).sum(axis=0)/(n-1)
		
		return var+epsilon#).sqrt()
	
	def _covar_offdiag(self,data,mu):

		n = data.shape[0]
		mask = torch.ones(self.latentDim,self.latentDim) - torch.eye(self.latentDim)
		covar = (data - mu).T @ (data - mu)/(n-1)
		masked = (covar * mask.to(self.device))**2 

		return masked

	def update_batch_covar(self,data):
		#print(data.shape)

		newCovar = data @ data.T / self.latentDim
		#print(newCovar.shape)
		#print(self.batch_approx.shape)

		updated = (1 - self.epsilon) *self.batch_approx.detach() + self.epsilon * newCovar

		assert updated.requires_grad == True

		self.batch_approx = updated
		return updated
	
	def gradMu_regularizer(self,z1):

		jac = vmap(jacrev(self.sde.MLP))(z1)
		reg = torch.logdet(jac)

		return reg.sum()

	def mu_regularizer(self,mu):

		return torch.linalg.vector_norm(mu,dim=1).sum()

	def kl_dim_only(self,dz,mu,Linv):

		"""
		current version: -lp - entropy_dims
		"""

		n = dz.shape[0]

		diff = (dz - mu)[:,:,None]
		transformedNormal = Linv @ (diff)

		empericalMeanDim = torch.nanmean(transformedNormal,axis=0).squeeze()
		empericalCovDim = (diff.squeeze() - empericalMeanDim).T @ (diff.squeeze() - empericalMeanDim)/(n-1)
		#const_dim = self.latentDim/2 * (np.log(2*np.pi) + 1)
		#det_dim = torch.logdet(empericalCovDim)/2
		#entropy_dim = const_dim + det_dim
		
		# E[log p] = -k/2 log (2pi) - 1/2 log | \Sigma| - 1/2 (x-\mu)^T \Sigma^{-1}(x-\mu)
		
		#lp = n*(-self.latentDim/2 * np.log(2*np.pi)) - (1/2 *(transformedNormal @ transformedNormal.transpose(-2,-1)).sum())

		#empericalCov = empericalCov + torch.eye(self.latentDim).to(self.device)*EPS
		#kl = -lp - entropy_dim
		kl = (empericalMeanDim **2).sum() - self.latentDim + \
			  torch.trace(empericalCovDim) - torch.logdet(empericalCovDim)

		return kl*n
	
	def entropy_loss(self,batch): #,dt=1):
		"""
		converting to entropy of batch, as opposed to entropy of
		latent distribution across dims
		"""

		n = batch.shape[0]
		mu = torch.mean(batch,axis=0,keepdim=True)
		cov = (batch-mu).T @ (batch-mu)/(self.latentDim-1)
		const = self.latentDim/2 * (np.log(2*np.pi) + 1)
		det = torch.logdet(cov)/2

		return  n*(det + const) #(det + const)
	


	def encode_trajectory(self,data):
		self.eval()

		return self.encoder.forward(data.to(self.device)).detach().cpu().numpy()
	
	def init_sde(self, loader,sde_optimizer,grad_clipper=None,n_epochs=100):

		self.train()
		epoch_losses = [] 

		vL = 0.
		lP = 0.
		cVL = 0.
		batchInd = np.random.choice(len(loader),1)

		rp = torch.randn((self.dataDim,self.latentDim)).to(self.device)
		for jj in tqdm(range(n_epochs),desc='initializing sde'):
			epoch_loss = 0.
			for ii,batch in enumerate(loader):

				sde_optimizer.zero_grad()

				x1,x2,dt = batch 
				x1,x2 = x1.to(self.device),x2.to(self.device)
				
				if len(x1.shape) > 2:
					x1,x2 = x1.view(x1.shape,-1),x2.view(x2.shape,[-1])
				
				z1,z2 = x1 @ rp, x2 @ rp 

				lp,mu,d = self.sde.loss(z1,z2,dt)
				
				lp.backward()
				if grad_clipper != None:
					grad_clipper(self.parameters())
				epoch_loss += lp.item()
				
				lP += lp.item()
				
				sde_optimizer.step()

			epoch_losses.append(epoch_loss/len(loader))
			assert epoch_loss != torch.nan, print('how?')
			self.sde.writer.add_scalar('Pre-train/lp',epoch_loss/len(loader),self.sde.epoch)

		#self.sde.writer.add_scalar('Train/covar loss',cVL/len(loader),self.sde.epoch)

			self.sde.epoch += 1

		sde_optimizer.zero_grad()
		return epoch_losses,sde_optimizer
	
	def generate_trajectory(self,init_conditions,T,dt):
		self.eval()
		init_conditions = init_conditions.to(self.device)
		if init_conditions.shape[-1] == self.dataDim:

			with torch.no_grad():
				z0 = self.encoder.forward(init_conditions)
			z0 = z0.squeeze().detach().cpu().numpy()

		elif init_conditions.shape[-1] == self.latentDim:

			z0 = init_conditions.detach().cpu().numpy()

		else:

			print(f"""initial conditions are {init_conditions.shape[-1]} dimensional, 
				  should be either {self.latentDim} or {self.dataDim} dimensional""")
			
			raise AssertionError
		
		traj = self.sde.generate(z0,T,dt)

		return traj
	
	def forward(self,batch,encode_grad=True,sde_grad=True,stopgrad=True,mode='kl'):
		
		x1,x2,dt = batch
		x1,x2,dt = x1.to(self.device),x2.to(self.device),dt.to(self.device)

		if stopgrad:
			z1,z2 = self.encoder.forward(x1),self.encoder.forward(x2,pass_gradient=False)
		elif encode_grad:
			z1,z2 = self.encoder.forward(x1),self.encoder.forward(x2)
		else:
			with torch.no_grad():
				z1,z2 = self.encoder.forward(x1),self.encoder.forward(x2)

		if sde_grad:
			lp,mu,d = self.sde.loss(z1,z2,dt)
		else:
			with torch.no_grad():

				lp,mu,d = self.sde.loss(z1,z2,dt)

		dz = z2 - z1
		zs = torch.vstack([z1,z2]) # bsz x latent dim
		#varLoss,covarLoss,muLoss = torch.zeros([0]),self.covar_loss(zs),torch.zeros([0])#0,0,0#self.var_loss(zs),self.covar_loss(zs),self.mu_reg(zs) #+ self.var_loss(z2)
		entropy = torch.zeros([0])#0#,self.entropy_loss(zs)
		if self.training:
			# only update batch covariance on training data
			currCovar = self.update_batch_covar(dz)
		#entropy_dz = self.entropy_loss(dz,dt=dt[0])

		
		#entropy_dz = self.entropy_loss_sumbatch(z2 - z1,dt=dt[0])
		#varLoss = self.snr_loss(zs) 
		if mode == 'kl':
			kl_loss = self.entropy_loss(dz)
			loss = -kl_loss #+ lp#lp - entropy_dz + self.mu*muLoss#+ self.mu * (varLoss + covarLoss) + muLoss #self.mu * varLoss
		elif mode == 'lp':
			loss = lp 
			kl_loss = self.entropy_loss(dz)
		elif mode == 'both':
			kl_loss = self.entropy_loss(dz)
			loss = lp - self.mu * kl_loss

		elif mode == 'kllp_gradmu':
			kl_loss = self.entropy_loss(dz)
			gradmu = self.gradMu_regularizer(z1)
			if torch.any(gradmu == torch.nan):
				print('we have nans in grad mu logdet')
			self.sde.writer.add_scalar('Train/gradmu',gradmu,self.counter)
			self.counter += 1
			loss = lp - kl_loss + self.mu*gradmu
		elif mode == 'kllp_mu':
			kl_loss = self.entropy_loss(dz)
			gradmu = self.mu_regularizer(mu)
			loss = lp - self.mu * kl_loss + gradmu

		elif mode == 'residuals_constrained':
			kl_loss = self.kl_dim_only(dz,mu,d)
			loss = kl_loss
		elif mode == 'allspace_constrained':
			kl_loss = self.kl_dim_only(dz,mu,d)
			entropy_dz = self.entropy_loss(dz)
			loss = lp + kl_loss - entropy_dz
		elif mode == 'batchCovar':
			kl_loss = self.kl_dim_only(dz,mu,d)
			batch_ld = torch.logdet(currCovar)/2

			assert kl_loss != torch.nan, print('kl loss is nan')
			assert batch_ld != torch.nan, print('logdet is nan')
			assert lp != torch.nan,print('log prob is nan')
			loss = lp +kl_loss - batch_ld
			assert loss != torch.nan, print('loss is somehow nan')

		else:
			raise Exception("Mode must be one of ['kl', 'lp', 'both']")
		
		return loss,z1,z2,mu,d,kl_loss,lp

	def train_epoch(self,loader,optimizer,grad_clipper=None,encode_grad=True,sde_grad=True,stopgrad=False,mode='kl'):

		self.train()
		epoch_loss = 0.
		epoch_mus = []
		epoch_Ds = []
		vL = 0.
		lP = 0.
		batchInd = np.random.choice(len(loader),1)

		for ii,batch in enumerate(loader):
			optimizer.zero_grad()
			loss,z1,z2,mu,d,vl,lp = self.forward(batch,encode_grad,sde_grad,stopgrad,mode=mode)
			assert loss != torch.nan, print('loss is somehow nan')

			loss.backward()
			if grad_clipper != None:
				grad_clipper(self.parameters())
			epoch_loss += loss.item()
			vL += vl.item()
			lP += lp.item()
			
			optimizer.step()
			
			epoch_mus.append(mu.detach().cpu().numpy())
			epoch_Ds.append(d.detach().cpu().numpy())


			if (ii == batchInd) & (self.sde.epoch % 100 ==0) & self.sde.plotDists:
				
				self._add_quiver(z1.detach().cpu().numpy(),mu.detach().cpu().numpy(),self.sde.p1name,'Train')
		epoch_mus = np.vstack(epoch_mus)
		epoch_Ds = np.vstack(epoch_Ds)

		assert epoch_loss != torch.nan, print('how?')
		self.sde.writer.add_scalar('Train/loss',epoch_loss/len(loader),self.sde.epoch)
		self.sde.writer.add_scalar('Train/KL',vL/len(loader),self.sde.epoch)
		self.sde.writer.add_scalar('Train/log prob',lP/len(loader),self.sde.epoch)
		#self.sde.writer.add_scalar('Train/covar loss',cVL/len(loader),self.sde.epoch)
		if self.sde.plotDists & (self.sde.epoch % 100 == 0):
			
			for d in range(self.sde.dim):
				
				self._add_dist_figure(epoch_mus[:,d],self.sde.p1name,d+1,'Train')
				#self.sde.writer.add_scalars(f'Train/{self.sde.p2name} dim {d+1}',{'estimated':epoch_sigs[d],'true':self.sde.true2[d]},self.sde.epoch)
			for d in range(self.sde.n_entries):
				self._add_dist_figure(epoch_Ds[:,d],self.sde.p2name,d+1,'Train')
		self.sde.epoch += 1
		optimizer.zero_grad()
		return epoch_loss,optimizer
	

	def val_epoch(self,loader):

		self.eval()
		with torch.no_grad():
			epoch_loss = 0.
			epoch_vl = 0.
			epoch_lp = 0.
			epoch_mus = []
			epoch_Ds = []

			for ii,batch in enumerate(loader):
				loss,z1,z2,mu,d,vl,lp = self.forward(batch)
				
				epoch_loss += loss.item()
				epoch_vl += vl.item()
				epoch_lp += lp.item()
				epoch_mus.append(mu.detach().cpu().numpy())
				epoch_Ds.append(d.detach().cpu().numpy())

				
		epoch_mus = np.vstack(epoch_mus)
		epoch_Ds = np.vstack(epoch_Ds)


		self.sde.writer.add_scalar('Test/loss',epoch_loss/len(loader),self.sde.epoch)
		self.sde.writer.add_scalar('Test/KL',epoch_vl/len(loader),self.sde.epoch)
		self.sde.writer.add_scalar('Test/log prob',epoch_lp/len(loader),self.sde.epoch)
		#self.sde.writer.add_scalar('Test/covar loss',epoch_cVL/len(loader),self.sde.epoch)

		return epoch_loss

	def test_epoch(self,loader):

		self.eval()
		with torch.no_grad():
			epoch_loss = 0.
			epoch_vl = 0.
			epoch_lp = 0.
			epoch_mus = []
			epoch_Ds = []

			batchInd = np.random.choice(len(loader))
			for ii,batch in enumerate(loader):
				loss,z1,z2,mu,d,vl,lp = self.forward(batch)
				
				epoch_loss += loss.item()
				epoch_vl += vl.item()
				epoch_lp += lp.item()

				epoch_mus.append(mu.detach().cpu().numpy())
				epoch_Ds.append(d.detach().cpu().numpy())

				if (ii == batchInd) & self.sde.plotDists:
					self._add_quiver(z1.detach().cpu().numpy(),mu.detach().cpu().numpy(),self.sde.p1name,'Test')

		epoch_mus = np.vstack(epoch_mus)
		epoch_Ds = np.vstack(epoch_Ds)

		self.sde.writer.add_scalar('Test/loss',epoch_loss/len(loader),self.sde.epoch)
		self.sde.writer.add_scalar('Test/KL',epoch_vl/len(loader),self.sde.epoch)
		self.sde.writer.add_scalar('Test/log prob',epoch_lp/len(loader),self.sde.epoch)
		#self.sde.writer.add_scalar('Test/covar loss',epoch_cVL/len(loader),self.sde.epoch)
		if self.sde.plotDists:
			for d in range(self.sde.dim):
				self._add_dist_figure(epoch_mus[:,d],self.sde.p1name,d+1,'Test')
				#self.sde.writer.add_scalars(f'Train/{self.sde.p2name} dim {d+1}',{'estimated':epoch_sigs[d],'true':self.sde.true2[d]},self.sde.epoch)
			for d in range(self.sde.n_entries):
				self._add_dist_figure(epoch_Ds[:,d],self.sde.p2name,d+1,'Test')
		"""
		for d in range(self.sde.dim):
			if self.sde.plotDists:
				self.sde.writer.add_scalars(f'Test/{self.sde.p1name} dim {d+1}',{'estimated':epoch_mus[d],'true':self.sde.true1[d]},self.sde.epoch)
				self.sde.writer.add_scalars(f'Test/{self.sde.p2name} dim {d+1}',{'estimated':epoch_sigs[d],'true':self.sde.true2[d]},self.sde.epoch)
			else:
				self.sde.writer.add_scalar(f'Test/{self.sde.p1name} dim {d+1}',epoch_mus[d],self.sde.epoch)
				self.sde.writer.add_scalar(f'Test/{self.sde.p2name} dim {d+1}',epoch_sigs[d],self.sde.epoch)
		"""
		return epoch_loss
	
	def save(self):

		sd = self.state_dict()
		fn = os.path.join(self.sde.save_dir,f'checkpoint_{self.sde.epoch}.tar')
		torch.save({
			'epoch': self.sde.epoch,
			'model_state_dict':sd
		},fn)
		return 
	
	def load(self,fn):
		print(f"Loading state from: {fn}")
		check = torch.load(fn)#,map_location=self.device)
		self.sde.epoch = check['epoch']
		self.load_state_dict(check['model_state_dict'])
		return
	
