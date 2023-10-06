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
		self.to(self.device)


	def encode_trajectory(self,data):

		return self.encoder.forward(data.to(self.device)).detach().cpu().numpy()
	
	def generate_trajectory(self,init_conditions,T,dt):

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
	
	def forward(self,batch):
		
		x1,x2,dt = batch
		x1,x2,dt = x1.to(self.device),x2.to(self.device),dt.to(self.device)

		z1,z2 = self.encoder.forward(x1),self.encoder.forward(x2,pass_gradient=False)

		loss,mu,d = self.sde.loss(z1,z2,dt)

		return loss,z1,z2,mu,d

	def train_epoch(self,loader,optimizer,grad_clipper=None):

		self.train()
		epoch_loss = 0.
		epoch_mus = []
		epoch_Ds = []
		true_p1 = []
		true_p2 = []
		batchInd = np.random.choice(len(loader),1)

		for ii,batch in enumerate(loader):

			loss,z1,z2,mu,d = self.forward(batch)
			loss.backward()
			if grad_clipper != None:
				grad_clipper(self.parameters())
			epoch_loss += loss.item()

			optimizer.step()
			epoch_mus.append(mu.detach().cpu().numpy())
			epoch_Ds.append(d.detach().cpu().numpy())
			
			true_p1.append(self.sde.true1(z1))
			true_p2.append(self.sde.true2(z1))


			if (ii == batchInd) & (self.sde.epoch % 100 ==0) & self.sde.plotDists:
				
				self.sde._add_quiver(z1.detach().cpu().numpy(),mu.detach().cpu().numpy(),self.sde.true1(z1),self.sde.p1name,'Train')
		epoch_mus = np.vstack(epoch_mus)
		epoch_Ds = np.vstack(epoch_Ds)
		true_p1 = np.vstack(true_p1)
		true_p2 = np.vstack(true_p2)

		self.sde.writer.add_scalar('Train/loss',epoch_loss/len(loader),self.sde.epoch)
		if self.sde.plotDists & (self.sde.epoch % 50 == 0):
			
			for d in range(self.sde.dim):
				
				self.sde._add_dist_figure(epoch_mus[:,d],true_p1[:,d],self.sde.p1name,d+1,'Train')
				#self.sde.writer.add_scalars(f'Train/{self.sde.p2name} dim {d+1}',{'estimated':epoch_sigs[d],'true':self.sde.true2[d]},self.sde.epoch)
			for d in range(self.sde.n_entries):
				self.sde._add_dist_figure(epoch_Ds[:,d],true_p2[:,d],self.sde.p2name,d+1,'Train')
		self.sde.epoch += 1
	
		return epoch_loss,optimizer
	

	def val_epoch(self,loader):

		self.eval()
		with torch.no_grad():
			epoch_loss = 0.
			epoch_mus = []
			epoch_Ds = []
			true_p1 = []
			true_p2 = []
			batchInd = np.random.choice(len(loader))
			for ii,batch in enumerate(loader):
				loss,z1,z2,mu,d = self.forward(batch)
				
				epoch_loss += loss.item()

				epoch_mus.append(mu.detach().cpu().numpy())
				epoch_Ds.append(d.detach().cpu().numpy())
				true_p1.append(self.sde.true1(z1))
				true_p2.append(self.sde.true2(z1))
				
		epoch_mus = np.vstack(epoch_mus)
		epoch_Ds = np.vstack(epoch_Ds)
		true_p1 = np.vstack(true_p1)
		true_p2 = np.vstack(true_p2)

		self.sde.writer.add_scalar('Test/loss',epoch_loss/len(loader),self.sde.epoch)

		return epoch_loss

	def test_epoch(self,loader):

		self.eval()
		with torch.no_grad():
			epoch_loss = 0.
			epoch_mus = []
			epoch_Ds = []
			true_p1 = []
			true_p2 = []
			batchInd = np.random.choice(len(loader))
			for ii,batch in enumerate(loader):
				loss,z1,z2,mu,d = self.forward(batch)
				
				epoch_loss += loss.item()

				epoch_mus.append(mu.detach().cpu().numpy())
				epoch_Ds.append(d.detach().cpu().numpy())
				true_p1.append(self.sde.true1(z1))
				true_p2.append(self.sde.true2(z1))
				if (ii == batchInd) & self.sde.plotDists:
					self.sde._add_quiver(z1.detach().cpu().numpy(),mu.detach().cpu().numpy(),self.sde.true1(z1),self.sde.p1name,'Test')

		epoch_mus = np.vstack(epoch_mus)
		epoch_Ds = np.vstack(epoch_Ds)
		true_p1 = np.vstack(true_p1)
		true_p2 = np.vstack(true_p2)

		self.sde.writer.add_scalar('Test/loss',epoch_loss/len(loader),self.sde.epoch)
		if self.sde.plotDists:
			for d in range(self.sde.dim):
				self.sde._add_dist_figure(epoch_mus[:,d],true_p1[:,d],self.sde.p1name,d+1,'Test')
				#self.sde.writer.add_scalars(f'Train/{self.sde.p2name} dim {d+1}',{'estimated':epoch_sigs[d],'true':self.sde.true2[d]},self.sde.epoch)
			for d in range(self.sde.n_entries):
				self.sde._add_dist_figure(epoch_Ds[:,d],true_p2[:,d],self.sde.p2name,d+1,'Test')
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