from latent_sde import *
from encoding import *
from tqdm import tqdm
from torch.func import jacrev,vmap
import seaborn as sns

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
		print("bugs cooked o7")

	def _init_batch_covar_approx(self,latent_size=512,epsilon = 0.2):

		"""
		create our first estimate of the batch covariance!
		
		initializes as the z x z identity, to begin with
		"""

		self.batch_size= latent_size
		self.batch_approx = torch.eye(latent_size).to(self.device)
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

	def __add_covar(self):

		fig1 = plt.figure(figsize=(10,10))
		ax = plt.gca()
		sns.heatmap(self.batch_approx.detach().cpu().numpy(),annot=True,fmt=".2f")
		self.sde.writer.add_figure(f'train/batch_approx', fig1,close=True,global_step=self.sde.epoch)

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

	def update_batch_covar_deprecated(self,data):
		#print(data.shape)

		newCovar = data @ data.T / self.latentDim
		#print(newCovar.shape)
		#print(self.batch_approx.shape)

		updated = (1 - self.epsilon) *self.batch_approx + self.epsilon * newCovar

		assert updated.requires_grad == True

		self.batch_approx = updated.detach()
		assert updated.requires_grad == True

		return updated
	
	def gradMu_regularizer(self,z1):

		jac = vmap(jacrev(self.sde.MLP))(z1)
		reg = torch.linalg.vector_norm(jac.view(z1.shape[0],-1),dim=1) #torch.logdet(jac)

		return reg.sum()

	def mu_regularizer(self,mu):

		return torch.linalg.vector_norm(mu,dim=1).sum()
	
	def kl_sde_encoder(self,mu_s,prec_s,mu_e,cov_e):

		"""
		inputs: 
		mu_s: torch.tensor, means of sde distribution
		prec_s: torch.tensor, precisions of sde distribution
		mu_e: torch.tensor, means of encoder distribution
		cov_e: covariance of encoder distribution

		returns:
			kl: sum of kls for each data point's distribution
		code is for 
		kl(encoder [e]||sde [s]) 
		= 1/2 [log |cov_s|/|cov_e| - k - 
		(mu_e - mu_s)^T(prec_s)(mu_e - mu_s) +
		tr{prec_s cov_q}]
		since det(A^-1) = 1/det(A)
		= 1/2 [-log |prec_s| - log |cov_e| - k - 
		(mu_e - mu_s)^T(prec_s)(mu_e - mu_s) +
		tr{prec_s cov_q}]

		"""

		mu_s,mu_e = mu_s.view(mu_s.shape[0],-1,1),mu_e.view(mu_s.shape[0],-1,1)	

		t1 = -torch.logdet(prec_s) - torch.logdet(cov_e)
		t2 = -self.latentDim
		t3 = (mu_e - mu_s).transpose(-2,-1) @ prec_s @ (mu_e - mu_s)
		t4 = torch.func.vmap(torch.trace)(prec_s @ cov_e)

		return (t1 + t2 + t3 + t4).sum()


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
		THIS IS SHOULD BE THE ENTROPY ACROSS THE DIMENSIONS!!!!!!!!!!!!!!!!
		THIS WAS BUGGED!!! MIS-SCALED!!! TRY THIS AGAIN!!!!!!
		"""

		n = batch.shape[0]
		mu = torch.mean(batch,axis=0,keepdim=True)
		cov = (batch-mu).T @ (batch-mu)/(n-1)
		const = self.latentDim/2 * (np.log(2*np.pi) + 1)
		det = torch.logdet(cov)/2

		return  n*(det + const) #(det + const)
	
	def update_batch_covar(self,covMat):

		updated = (1 - self.epsilon) *self.batch_approx.detach() + self.epsilon * covMat

		assert updated.requires_grad == True

		self.batch_approx = updated
		return updated

	def entropy_loss_ma(self,batch):
		"""
		same as the previous function, but now we're using a
		moving average to calculate the batch entropy -- amortizing
		across all our batches
		"""

		n  = batch.shape[0]
		mu = torch.mean(batch,axis=0,keepdim=True)
		cov = (batch-mu).T @ (batch-mu)/(n-1)
		const = self.latentDim/2 * (np.log(2*np.pi) + 1)

		updated = self.update_batch_covar(cov)

		det = torch.logdet(updated)/2

		return  n*(det + const)


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
		
		if len(batch) == 3:
			x1,x2,dt = batch
			x1,x2,dt = x1.to(self.device),x2.to(self.device),dt.to(self.device)
		else:
			x1,x2,x3,dt = batch
			x1,x2,x3,dt = x1.to(self.device),x2.to(self.device),x3.to(self.device),dt.to(self.device)


		z1,z2 = self.encoder.forward(x1),self.encoder.forward(x2)
		if len(batch) == 4:
			z3 = self.encoder.forward(x3)
		"""
		if stopgrad:
			if self.encoder.type == 'deterministic':
				z1,z2 = self.encoder.forward(x1),self.encoder.forward(x2,pass_gradient=False)
				if len(batch) == 4:
					z3 = self.encoder.forward(x3)
			else:
				(z1,cov1), (z2,cov2) = self.encoder.forward(x1,type='prob'),self.encoder.forward(x2,pass_gradient=False,type='prob')
		elif encode_grad:
			if self.encoder.type == 'deterministic':
				z1,z2 = self.encoder.forward(x1),self.encoder.forward(x2)
				if len(batch) == 4:
					z3 = self.encoder.forward(x3)
			else:
				(z1,cov1), (z2,cov2) = self.encoder.forward(x1,type='prob'),self.encoder.forward(x2,type='prob')
		else:
			with torch.no_grad():
				if self.encoder.type == 'deterministic':
					z1,z2 = self.encoder.forward(x1),self.encoder.forward(x2)
					if len(batch) == 4:
						z3 = self.encoder.forward(x3)
				else:
					(z1,cov1), (z2,cov2) = self.encoder.forward(x1,type='prob'),self.encoder.forward(x2,type='prob')
		"""
		lp,mu,d = self.sde.loss(z1,z2,dt)
		mu2 = self.sde.MLP(z2)			

		dz = z2 - z1
		if len(batch) == 4:
			l2,_,_ = self.sde.loss(z2,z3,dt)
			lp += l2

			dz2 = z3 - z2
			#linLoss = self.mu * self._linearity_penalty(dz,dz2)
			#dz = torch.vstack([dz,dz2])
			kl_loss = self.entropy_loss(torch.vstack([dz,dz2]))
			linLoss = self.mu *self._linearity_penalty(dz,dz2)
		else:
			linLoss = self.mu *self._linearity_penalty(mu,mu2)
			kl_loss = self.entropy_loss(dz)
		assert not torch.isnan(lp)
		
		zs = torch.vstack([z1,z2]) # bsz x latent dim
		#varLoss,covarLoss,muLoss = torch.zeros([0]),self.covar_loss(zs),torch.zeros([0])#0,0,0#self.var_loss(zs),self.covar_loss(zs),self.mu_reg(zs) #+ self.var_loss(z2)
		entropy = torch.zeros([0])#0#,self.entropy_loss(zs)
		#if self.training:
			# only update batch covariance on training data
		#	currCovar = self.update_batch_covar(dz)
		#entropy_dz = self.entropy_loss(dz,dt=dt[0])

		
		#entropy_dz = self.entropy_loss_sumbatch(z2 - z1,dt=dt[0])
		#varLoss = self.snr_loss(zs) 
			
		if mode == 'kl':
			#kl_loss = self.entropy_loss(dz)
			loss = -kl_loss #+ lp#lp - entropy_dz + self.mu*muLoss#+ self.mu * (varLoss + covarLoss) + muLoss #self.mu * varLoss
		elif mode == 'probkl':
			print("Don't use this")
			#assert self.encoder.type == 'probabilistic', print("This loss needs a probabilistic encoder!!!")
			#kl_loss = self.kl_sde_encoder(z1 + mu,d @ d.transpose(-2,-1),z2,cov2)
			loss = kl_loss
		elif mode == 'lp':
			loss = lp 
			#kl_loss = self.entropy_loss(dz)
		elif mode == 'both':
			#kl_loss = self.entropy_loss(dz)
			loss = lp - kl_loss
		elif mode == 'linearityTest':
			#kl_loss = self.entropy_loss(dz)
			
			loss = lp - kl_loss - linLoss

		elif mode == 'both_ma':
			kl_loss = self.entropy_loss_ma(dz)
			loss = lp - kl_loss

		elif mode == 'kllp_gradmu':
			#kl_loss = self.entropy_loss(dz)
			gradmu = self.gradMu_regularizer(z1)
			if torch.any(gradmu == torch.nan):
				print('we have nans in grad mu logdet')
			self.sde.writer.add_scalar('Train/gradmu',gradmu,self.counter)
			self.counter += 1
			loss = lp - kl_loss + gradmu
		elif mode == 'kllp_mu':
			#kl_loss = self.entropy_loss(dz)
			gradmu = self.mu_regularizer(mu)
			loss = lp - kl_loss + gradmu

		elif mode == 'residuals_constrained':
			kl_loss = self.kl_dim_only(dz,mu,d)
			loss = kl_loss
		elif mode == 'allspace_constrained':
			kl_loss = self.kl_dim_only(dz,mu,d)
			entropy_dz = self.entropy_loss(dz)
			loss = lp + kl_loss - entropy_dz

		else:
			raise Exception("Mode must be one of ['kl', 'lp', 'both']")
		
		return loss,z1,z2,linLoss,d,kl_loss,lp


	def _normalize_grads(self,norm_const,normPart = 'encoder'):

		if normPart == 'encoder':

			for param in self.encoder.parameters():
				
				param.grad /= norm_const 

		else:
			for param in self.sde.parameters():
				param.grad /= norm_const

		return 
	
	def _linearity_penalty(self,f1,f2):
		"""
		returns negative cosine similarity between successive drift terms
		(encouraging them to be locally aligned/linear). This assumes we are
		minimizing a function, rather than maximizing it
		"""

		dotProd = (f1 * f2).sum(dim=-1)
		return  (dotProd/ (torch.norm(f1,dim=-1) * torch.norm(f2,dim=-1) + EPS)).mean()
	
	def e_step(self,loader,embedopt,grad_clipper=None):
		self.train()
		epoch_loss = 0.
		epoch_mus = []
		epoch_Ds = []
		vL = 0.
		lP = 0.
		embedopt.zero_grad()
		for ii,batch in enumerate(loader):
			
			loss,z1,z2,mu,d,vl,lp = self.forward(batch,encode_grad=True,sde_grad=True,stopgrad=False,mode='both')
			assert loss != torch.nan, print('loss is somehow nan')

			loss.backward()
			if grad_clipper != None:
				grad_clipper(self.parameters())
			epoch_loss += loss.item()
			vL += vl.item()
			lP += lp.item()
						
			epoch_mus.append(mu.detach().cpu().numpy())
			epoch_Ds.append(d.detach().cpu().numpy())

		self._normalize_grads(len(loader),normPart='encoder')
		embedopt.step()
		epoch_mus = np.vstack(epoch_mus)
		epoch_Ds = np.vstack(epoch_Ds)

		assert epoch_loss != torch.nan, print('how?')
		#if mode == 'both_ma':
		#	self.__add_covar()
			#self.sde.writer.add_image('approximate covar',self.batch_approx.view(1,*self.batch_approx.shape),self.sde.epoch)

		self.sde.writer.add_scalar('Train/loss',epoch_loss/len(loader),self.sde.epoch)
		self.sde.writer.add_scalar('Train/KL',vL/len(loader),self.sde.epoch)
		self.sde.writer.add_scalar('Train/log prob',lP/len(loader),self.sde.epoch)
		self.sde.epoch += 1
		embedopt.zero_grad()
		return epoch_loss,embedopt

	def m_step(self,loader,sdeopt,grad_clipper=None):
		self.train()
		epoch_loss = 0.
		epoch_mus = []
		epoch_Ds = []
		vL = 0.
		lP = 0.
		for ii,batch in enumerate(loader):
			sdeopt.zero_grad()
			loss,z1,z2,mu,d,vl,lp = self.forward(batch,encode_grad=True,sde_grad=True,stopgrad=False,mode='both')
			assert loss != torch.nan, print('loss is somehow nan')

			loss.backward()
			if grad_clipper != None:
				grad_clipper(self.parameters())
			epoch_loss += loss.item()
			vL += vl.item()
			lP += lp.item()
			
			sdeopt.step()
			
			epoch_mus.append(mu.detach().cpu().numpy())
			epoch_Ds.append(d.detach().cpu().numpy())

		epoch_mus = np.vstack(epoch_mus)
		epoch_Ds = np.vstack(epoch_Ds)

		assert epoch_loss != torch.nan, print('how?')
		#if mode == 'both_ma':
		#	self.__add_covar()
			#self.sde.writer.add_image('approximate covar',self.batch_approx.view(1,*self.batch_approx.shape),self.sde.epoch)

		self.sde.writer.add_scalar('Train/loss',epoch_loss/len(loader),self.sde.epoch)
		self.sde.writer.add_scalar('Train/KL',vL/len(loader),self.sde.epoch)
		self.sde.writer.add_scalar('Train/log prob',lP/len(loader),self.sde.epoch)
		self.sde.epoch += 1
		sdeopt.zero_grad()
		return epoch_loss,sdeopt

	def train_epoch_accum_grad(self,loader,sdeopt,embedopt,grad_clipper=None,encode_grad=True,sde_grad=True,stopgrad=False,mode='both'):

		self.train()
		epoch_loss = 0.
		epoch_mus = []
		epoch_Ds = []
		vL = 0.
		lP = 0.
		ll = 0.
		batchInd = np.random.choice(len(loader),1)
		embedopt.zero_grad()
		for ii,batch in enumerate(loader):
			sdeopt.zero_grad()
			loss,z1,z2,linloss,d,vl,lp = self.forward(batch,encode_grad,sde_grad,stopgrad,mode=mode)
			assert not torch.isnan(loss), print('loss is somehow nan')

			loss.backward()
			if grad_clipper != None:
				grad_clipper(self.parameters())
			epoch_loss += loss.item()
			vL += vl.item()
			lP += lp.item()
			ll += linloss.item()
			
			sdeopt.step()
			
			#epoch_mus.append(mu.detach().cpu().numpy())
			#epoch_Ds.append(d.detach().cpu().numpy())


			#if (ii == batchInd) & (self.sde.epoch % 100 ==0) & self.sde.plotDists:
				
			#	self._add_quiver(z1.detach().cpu().numpy(),mu.detach().cpu().numpy(),self.sde.p1name,'Train')

		self._normalize_grads(len(loader),normPart='encoder')
		embedopt.step()
		#epoch_mus = np.vstack(epoch_mus)
		#epoch_Ds = np.vstack(epoch_Ds)

		assert epoch_loss != torch.nan, print('how?')
		#if mode == 'both_ma':
		#	self.__add_covar()
			#self.sde.writer.add_image('approximate covar',self.batch_approx.view(1,*self.batch_approx.shape),self.sde.epoch)
		self.sde.writer.add_scalar('Train/linearity loss',ll/len(loader),self.sde.epoch)
		self.sde.writer.add_scalar('Train/loss',epoch_loss/len(loader),self.sde.epoch)
		self.sde.writer.add_scalar('Train/KL',vL/len(loader),self.sde.epoch)
		self.sde.writer.add_scalar('Train/log prob',lP/len(loader),self.sde.epoch)
		self.sde.epoch += 1
		sdeopt.zero_grad()
		embedopt.zero_grad()
		return epoch_loss,sdeopt,embedopt

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
		#if mode == 'both_ma':
		#	self.__add_covar()
			#self.sde.writer.add_image('approximate covar',self.batch_approx.view(1,*self.batch_approx.shape),self.sde.epoch)

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
	

	def val_epoch(self,loader,mode='both'):

		self.eval()
		with torch.no_grad():
			epoch_loss = 0.
			epoch_vl = 0.
			epoch_lp = 0.
			epoch_ll = 0
			epoch_mus = []
			epoch_Ds = []

			for ii,batch in enumerate(loader):
				loss,z1,z2,linloss,d,vl,lp = self.forward(batch,mode=mode)
				
				epoch_loss += loss.item()
				epoch_vl += vl.item()
				epoch_lp += lp.item()
				epoch_ll += linloss.item()
				#epoch_mus.append(mu.detach().cpu().numpy())
				#epoch_Ds.append(d.detach().cpu().numpy())

				
		#epoch_mus = np.vstack(epoch_mus)
		#epoch_Ds = np.vstack(epoch_Ds)

		self.sde.writer.add_scalar('Test/linearity loss',epoch_ll/len(loader),self.sde.epoch)
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
			epoch_ll = 0.
			epoch_mus = []
			epoch_Ds = []

			batchInd = np.random.choice(len(loader))
			for ii,batch in enumerate(loader):
				loss,z1,z2,linloss,d,vl,lp = self.forward(batch)
				
				epoch_loss += loss.item()
				epoch_vl += vl.item()
				epoch_lp += lp.item()
				epoch_ll += linloss.item()

				#epoch_mus.append(mu.detach().cpu().numpy())
				#epoch_Ds.append(d.detach().cpu().numpy())

				#if (ii == batchInd) & self.sde.plotDists:
				#	self._add_quiver(z1.detach().cpu().numpy(),mu.detach().cpu().numpy(),self.sde.p1name,'Test')

		#epoch_mus = np.vstack(epoch_mus)
		#epoch_Ds = np.vstack(epoch_Ds)

		self.sde.writer.add_scalar('Test/linearity loss',epoch_ll/len(loader),self.sde.epoch)
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
	
