import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from my_models import nonlinearLatentSDENatParams,nonlinearLatentSDE,Simple1dTestDE
from train import train,trainAlternating
from data import *
from tqdm import tqdm
import seaborn as sns
from plot import plotSamples1d,plotSamples3d,plot1dFlows,plot1dSigmas



if __name__ == '__main__':


	dt = 0.001
	xs = generate_geometric_brownian(512,dt=0.001,T = 1)
	xsTest = generate_geometric_brownian(512,dt=0.001,T=1)
	dls = makeToyDataloaders(xs,xsTest,sampledt=0.01,truedt=0.001)
	#linearmodel.load('/home/miles/test1_linear_middt/checkpoint_500.tar')
	#lrs = [5e-5, 1e-5,5e-6,1e-6]
	lr = 1e-5
	#for lr in lrs:
	linearmodel = Simple1dTestDE(save_dir=f'/home/miles/test1d_layout')
	linearmodel = train(linearmodel,dls,nEpochs=5000,save_freq=100,test_freq=25,lr=lr,step_size=None)

	linearmodel2 = nonlinearLatentSDENatParams(dim=1,diag_covar=True,save_dir=f'/home/miles/natparams_longrun_stepLR_{lr}_lr')
	linearmodel2 = train(linearmodel2,dls,nEpochs=5000,save_freq=100,test_freq=25,lr=1e-4,step_size=100)
	#linearmodel2.load('/home/miles/test2_linear_natparms/checkpoint_800.tar')
	print(f"generating new data! {1024} samples")
	
	samples = []
	muDist = []
	sigDist=[]

	samplesNat = []
	muDistNat = []
	sigDistNat =[]
	skip = int(0.01/0.001)
	for ii in tqdm(range(256), desc="generating trajectories"):
		x = xs[ii]
		x = torch.from_numpy(x).type(torch.FloatTensor).to(linearmodel.device)
		mus = linearmodel.MLP(x[:-1,:]).detach().cpu().numpy().squeeze()
		muDist.append(mus/x[:-1].detach().cpu().numpy().squeeze())
		Ds = np.exp(linearmodel.D(x[:-1]).detach().cpu().numpy().squeeze())
		sigDist.append(Ds/x[:-1].detach().cpu().numpy().squeeze())
		#plot1dFlows(x.detach().cpu().numpy()[:-1],mus * dt,dt=dt)
		#plot1dSigmas(x.detach().cpu().numpy()[:-1],Ds*np.sqrt(dt),dt=dt)
		#### TOO MCUH D+LINEAR ALGEMBRA
		chol = torch.zeros(x[:-1].shape[0],linearmodel2.dim,linearmodel2.dim)
		D = torch.exp(linearmodel2.D(x[:-1])).detach().cpu()
		chol[:,linearmodel2.tril_inds[0],linearmodel2.tril_inds[1]] = D
		etas = linearmodel2.MLP(x[:-1]).detach().cpu().view(x[:-1].shape[0],x.shape[1],1)
		eyes = torch.eye(linearmodel2.dim).view(1,linearmodel2.dim,linearmodel2.dim).repeat(x[:-1].shape[0],1,1)
		invChol = torch.linalg.solve_triangular(chol,eyes,upper=False)
		sigma = invChol.transpose(-2,-1) @ invChol 
		mus = sigma @ etas
		mus=mus.squeeze().numpy()
		#plot1dFlows(x.detach().cpu().numpy()[:-1],mus * dt,dt=dt)
		#plot1dSigmas(x.detach().cpu().numpy()[:-1],D.numpy()*np.sqrt(dt),dt=dt)
		muDistNat.append(mus/x[:-1].detach().cpu().numpy().squeeze())
		sigDistNat.append(invChol.numpy().squeeze()/x[:-1].detach().cpu().numpy().squeeze())
		#invChol = scipy.linalg.solve_triangular(chols,eyes,lower=True,check_finite=False)

		#steps = mus*dt + Ds*np.sqrt(dt)*np.random.randn(*(x[:-1].squeeze().shape))
		#steps = steps.detach().cpu().numpy()
		#xGen = x[:-1].detach().cpu().numpy().squeeze() + steps
		
		#samples.append(np.hstack([x[0],xGen.squeeze()]))
		
		x0 = 0.1 + 0.03**2 * np.random.randn(1)
		xsTmp = linearmodel.generate(x0,T=1,dt=0.01)
		samples.append(xsTmp)
		xsTmp = linearmodel2.generate(x0,T=1,dt=0.01)
		samplesNat.append(xsTmp)
	
	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	muDist = np.hstack(muDist)
	sigDist=np.hstack(sigDist)
	#print(len(np.hstack(muDist)))
	sns.kdeplot(x=muDist,ax=ax1)
	sns.kdeplot(x=sigDist,ax=ax2)
	ax1.set_xlabel("value of mu")
	ax2.set_xlabel("Values of sigma")
	ax1.set_title(f"{np.nanmedian(muDist)}")
	ax2.set_title(f"{np.nanmedian(sigDist)}")
	plt.show()
	plt.close(fig)

	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	muDistNat = np.hstack(muDistNat)
	sigDistNat=np.hstack(sigDistNat)
	#print(len(np.hstack(muDist)))
	sns.kdeplot(x=muDistNat,ax=ax1)
	sns.kdeplot(x=sigDistNat,ax=ax2)
	ax1.set_xlabel("value of mu")
	ax2.set_xlabel("Values of sigma")
	ax1.set_title(f"{np.nanmedian(muDistNat)}")
	ax2.set_title(f"{np.nanmedian(sigDistNat)}")
	plt.show()
	plt.close(fig)

	print("done!")
	plotSamples1d(xs,samples)
	plotSamples1d(xs,samplesNat)
   
	#plot_sde(xs)
	xs = generate_stochastic_lorenz(1024,dt=0.025/5,T = 1,coeffs=[10,28,8/3,.15,.15,.15])
	#plotSamples3d(xs,xs)
	model = nonlinearLatentSDE(dim=3,diag_covar=True,save_dir='/home/miles/test1_nonlinear')
	dls = makeToyDataloaders(np.vstack(xs),np.vstack(xs),dt=0.025/5)
	#model.load('/home/miles/test1/checkpoint_1000.tar')
	model = train(model,dls,nEpochs=2500,save_freq=500,test_freq=200)
	print(f"generating new data! {1000/.01} samples")
	samples = []
	for ii in range(1024):
		x0 = np.random.randn(3)
		xsTmp = model.generate(x0,T=1,dt=0.025/5)
		samples.append(xsTmp)
	
	print("done!")
	plotSamples3d(xs,samples)
	#sde_gif([xsTmp,xs[0]])
	#plot_sde(xs)
