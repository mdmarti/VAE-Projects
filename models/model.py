from latent_sde import *
from encoding import *


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
		self.to(self.device)

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
	
	def snr_loss(self,batch):

		mu = torch.mean(batch,axis=0)
		sd = torch.std(batch,axis=0)
		#mu2 = torch.pow(mu,2)
		sd2 = torch.pow(sd,2)
		t1 = mu/sd
		#t2 = 1 - mu/sd2
		t2 = sd2 - mu
		t2[t2 > 0] = 0
		t2 = self.k * t2

		assert t2.shape == t1.shape 

		return (t1 - t2).sum()
	
	def var_loss(self,batch,epsilon=1e-5):


		mu = batch.mean(axis=0,keepdims=True)
		sd = self._reg_sd(batch,mu,epsilon)
		hinge = self.gamma - sd.sum() # gamma = latentDim
		#inds = hinge < 0
		#hinge[inds] = 0.
		return hinge * (hinge >= 0)
	
	def encode_trajectory(self,data):
		self.eval()

		return self.encoder.forward(data.to(self.device)).detach().cpu().numpy()
	
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
	
	def forward(self,batch,encode_grad=True,sde_grad=True,stopgrad=True):
		
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

		zs = torch.vstack([z1,z2]) # bsz x latent dim
		varLoss = self.var_loss(zs) #+ self.var_loss(z2)
		#varLoss = self.snr_loss(zs) 
		loss = lp + self.mu * varLoss#self.mu * varLoss
		return loss,z1,z2,mu,d,varLoss,lp

	def train_epoch(self,loader,optimizer,grad_clipper=None,encode_grad=True,sde_grad=True,stopgrad=False):

		self.train()
		epoch_loss = 0.
		epoch_mus = []
		epoch_Ds = []
		vL = 0.
		lP = 0.
		batchInd = np.random.choice(len(loader),1)

		for ii,batch in enumerate(loader):

			loss,z1,z2,mu,d,vl,lp = self.forward(batch,encode_grad,sde_grad,stopgrad)
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


		self.sde.writer.add_scalar('Train/loss',epoch_loss/len(loader),self.sde.epoch)
		self.sde.writer.add_scalar('Train/variance loss',vL/len(loader),self.sde.epoch)
		self.sde.writer.add_scalar('Train/log prob',lP/len(loader),self.sde.epoch)
		if self.sde.plotDists & (self.sde.epoch % 100 == 0):
			
			for d in range(self.sde.dim):
				
				self._add_dist_figure(epoch_mus[:,d],self.sde.p1name,d+1,'Train')
				#self.sde.writer.add_scalars(f'Train/{self.sde.p2name} dim {d+1}',{'estimated':epoch_sigs[d],'true':self.sde.true2[d]},self.sde.epoch)
			for d in range(self.sde.n_entries):
				self._add_dist_figure(epoch_Ds[:,d],self.sde.p2name,d+1,'Train')
		self.sde.epoch += 1
	
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
		self.sde.writer.add_scalar('Test/variance loss',epoch_vl/len(loader),self.sde.epoch)
		self.sde.writer.add_scalar('Test/log prob',epoch_lp/len(loader),self.sde.epoch)

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
		self.sde.writer.add_scalar('Test/variance loss',epoch_vl/len(loader),self.sde.epoch)
		self.sde.writer.add_scalar('Test/log prob',epoch_lp/len(loader),self.sde.epoch)
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