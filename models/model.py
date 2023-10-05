from latent_sde import *
from encoding import *


class EmbeddingSDE(nn.Module):


	def __init__(self,dataDim,latentDim,encoding,latentSDE,
				 device='cuda'):


		self.sde = latentSDE
		self.encoder =encoding 
		self.dataDim=dataDim
		self.latentDim=latentDim 
		self.to(self.device)


	def encode_trajectory(self,data):

		return self.encoder.encode(data)
	
	def generate_trajectory(self,init_conditions,T,dt):

		if init_conditions.shape[-1] == self.dataDim:

			z0 = self.encoder.encode(init_conditions)

		elif init_conditions.shape[-1] == self.latentDim:

			z0 = init_conditions

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
			
			true_p1.append(self.sde.true1(batch[0]))
			true_p2.append(self.sde.true2(batch[0]))


			if (ii == batchInd) & (self.epoch % 100 ==0) & self.sde.plotDists:
				
				self.sde._add_quiver(batch[0].detach().cpu().numpy(),mu.detach().cpu().numpy(),self.true1(batch[0]),self.p1name,'Train')
		epoch_mus = np.vstack(epoch_mus)
		epoch_Ds = np.vstack(epoch_Ds)
		true_p1 = np.vstack(true_p1)
		true_p2 = np.vstack(true_p2)

		self.sde.writer.add_scalar('Train/loss',epoch_loss/len(loader),self.epoch)
		if self.plotDists & (self.epoch % 50 == 0):
			
			for d in range(self.dim):
				
				self.sde._add_dist_figure(epoch_mus[:,d],true_p1[:,d],self.p1name,d+1,'Train')
				#self.writer.add_scalars(f'Train/{self.p2name} dim {d+1}',{'estimated':epoch_sigs[d],'true':self.true2[d]},self.epoch)
			for d in range(self.n_entries):
				self.sde._add_dist_figure(epoch_Ds[:,d],true_p2[:,d],self.p2name,d+1,'Train')
		self.epoch += 1
	
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
				loss,mu,d = self.forward(batch)
				
				epoch_loss += loss.item()

				epoch_mus.append(mu.detach().cpu().numpy())
				epoch_Ds.append(d.detach().cpu().numpy())
				true_p1.append(self.true1(batch[0]))
				true_p2.append(self.true2(batch[0]))
				if (ii == batchInd) & self.plotDists:
					self._add_quiver(batch[0].detach().cpu().numpy(),mu.detach().cpu().numpy(),self.true1(batch[0]),self.p1name,'Test')

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