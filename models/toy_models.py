import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from my_models import nonlinearLatentSDENatParams,nonlinearLatentSDE,Simple1dTestDE
from train import train,trainAlternating
from data import *
from tqdm import tqdm
import seaborn as sns
from plot import plotSamples1d,plotSamples3d,plotSamples2d
import os

def run_geometric_brownian_experiment(): 

	dt = 0.001
	xs = generate_geometric_brownian(1024,dt=dt,T = 1)
	xsTest = generate_geometric_brownian(256,dt=dt,T=1)
	xsDownsampled = downsample(xs,origdt=dt,newdt=0.02,noise=False)
	xsTestDownsampled = downsample(xsTest,origdt=dt,newdt=0.02,noise=False)
	dls = makeToyDataloaders(xsDownsampled,xsTestDownsampled,dt=0.02)
	#linearmodel.load('/home/miles/test1_linear_middt/checkpoint_500.tar')
	lrs = [5e-5,1e-5,5e-6]#[1e-2, 1e-3]#
	lr = 5e-6
	print('decay tests: nat params 3')
	#for lr in lrs:
		#linearmodel = nonlinearLatentSDE(dim=1,save_dir=f'/home/miles/sde_models/test_fix_full_lr_{lr}')
		#linearmodel = train(linearmodel,dls,nEpochs=5000,save_freq=100,test_freq=25,lr=lr,gamma=0.999)

	#	linearmodel2 = nonlinearLatentSDENatParams(dim=1,save_dir=f'/home/miles/sde_models/testfix_natparams_lr_{lr}')
	#	linearmodel2 = train(linearmodel2,dls,nEpochs=5000,save_freq=100,test_freq=25,lr=lr,gamma=0.999)
	#linearmodel2.load('/home/miles/test2_linear_natparms/checkpoint_800.tar')
	linearmodel2 = nonlinearLatentSDENatParams(dim=1,save_dir=f'/home/miles/sde_models/test_fix_full_lr_{lr}')
	checkpointDir = f'/home/miles/sde_models/testfix_natparams_lr_{lr}'
	checkpoint = os.path.join(checkpointDir,'checkpoint_5000.tar')
	linearmodel2.load(checkpoint)
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
		x = torch.from_numpy(x).type(torch.FloatTensor).to(linearmodel2.device)
		#mus = linearmodel.MLP(x[:-1,:]).detach().cpu().numpy().squeeze()
		#muDist.append(mus/x[:-1].detach().cpu().numpy().squeeze())
		#Ds = np.exp(linearmodel.D(x[:-1]).detach().cpu().numpy().squeeze())
		#sigDist.append(Ds/x[:-1].detach().cpu().numpy().squeeze())
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
		#xsTmp = linearmodel.generate(x0,T=1,dt=0.01)
		#samples.append(xsTmp)
		xsTmp = linearmodel2.generate(x0,T=1,dt=0.01)
		samplesNat.append(xsTmp)
	"""
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
	"""
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
	#plotSamples1d(xs,samples)
	plotSamples1d(xs,samplesNat)

def run_stochastic_lorenz_experiment():

   #plot_sde(xs)
	dt = 0.001
	xs = generate_stochastic_lorenz(1024,dt=dt,T = 1,coeffs=[10,28,8/3,.15,.15,.15])
	xsTest = generate_stochastic_lorenz(1024,dt=dt,T = 1,coeffs=[10,28,8/3,.15,.15,.15])
	#plotSamples3d(xs,xs)
	xsDownsampled = downsample(xs,origdt=dt,newdt=0.025,noise=False)
	xsTestDownsampled = downsample(xsTest,origdt=dt,newdt=0.025,noise=False)
	model = nonlinearLatentSDE(dim=3,save_dir='/home/miles/test1_nonlinear',plotTrue=False)
	dls = makeToyDataloaders(xsDownsampled,xsTestDownsampled,dt=0.025)
	#model.load('/home/miles/test1/checkpoint_1000.tar')
	lr = 5e-6
	model = train(model,dls,nEpochs=2500,save_freq=500,test_freq=100,lr=lr,gamma=0.999)
	print(f"generating new data! {1000/.01} samples")
	samples = []
	for ii in range(1024):
		x0 = np.random.randn(3)
		xsTmp = model.generate(x0,T=1,dt=0.025/5)
		samples.append(xsTmp)
	
	print("done!")
	plotSamples3d(xs,samples)

def run_2d_swirly_boy():

	sampledt=0.001
	observeddt=0.02
	x = generate_2d_swirls(n=1024,dt=sampledt,T=1,\
			mu=1.1,theta=720*sampledt,sigma=0.5,x0=np.array([0.05,-0.05]))
	xDownsampled = downsample(x,origdt=sampledt,newdt=observeddt,noise=False)
	xTest = generate_2d_swirls(n=1024,dt=sampledt,T=1,\
			mu=1.1,theta=720*sampledt,sigma=0.5,x0=np.array([0.05,-0.05]))
	xTestDownsampled = downsample(xTest,origdt=sampledt,newdt=observeddt,noise=False)

	lr = 1e-5
	model1 = nonlinearLatentSDE(dim=2,save_dir=f'/home/miles/attempt_2d_one_lr_{lr}_diag',plotTrue=False,diag=True)
	dls = makeToyDataloaders(xDownsampled,xTestDownsampled,dt=observeddt)
	model2 = nonlinearLatentSDENatParams(dim=2,save_dir=f'/home/miles/attempt_2d_one_natparms_lr_{lr}_diag',plotTrue=False,diag=True)

	
	#model1 = train(model1,dls,nEpochs=5000,save_freq=500,test_freq=100,lr=lr,gamma=0.995)
	#model2 = train(model2,dls,nEpochs=5000,save_freq=500,test_freq=100,lr=lr,gamma=0.995)
	checkpointDir1 = f'/home/miles/attempt_2d_one_lr_{lr}_diag'
	checkpointDir2 = f'/home/miles/attempt_2d_one_natparms_lr_{lr}_diag'
	checkpoint1 = os.path.join(checkpointDir1,'checkpoint_5000.tar')
	checkpoint2 = os.path.join(checkpointDir2,'checkpoint_5000.tar')
	model1.load(checkpoint1)
	model2.load(checkpoint2)
	#print(f"generating new data! {1000/.01} samples")
	samples1 = []
	samples2 = []
	for ii in tqdm(range(256), desc="generating trajectories"):
		x0=np.array([0.05,-0.05]) +  0.03**2 * np.random.randn(2)
		xsTmp = model1.generate(x0,T=1,dt=sampledt)
		samples1.append(xsTmp)

		x0=np.array([0.05,-0.05]) +  0.03**2 * np.random.randn(2)
		xsTmp = model2.generate(x0,T=1,dt=sampledt)
		samples2.append(xsTmp)
	
	print("done!")
	plotSamples2d(x,samples1)
	plotSamples2d(x,samples2)

	return 

if __name__ == '__main__':

	#run_geometric_brownian_experiment()
	run_2d_swirly_boy()
	#run_stochastic_lorenz_experiment()
	
	
	#sde_gif([xsTmp,xs[0]])
	#plot_sde(xs)
