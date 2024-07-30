import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib import animation

EPSILON = 1e-12

def z_score(data):

	x_stacked =np.vstack(data)
	mu = np.nanmean(x_stacked,axis=0,keepdims=True)
	sd = np.nanstd(x_stacked,axis=0,keepdims=True)

	return [(d - mu)/sd for d in data],mu,sd

def scale(data):
	x_stacked =np.vstack(data)
	sd = np.nanstd(x_stacked,axis=0,keepdims=True)
	mag = np.amax(np.abs(x_stacked))
	return [d/mag for d in data],mag

def generate_vanderpol(n=100,T = 1, dt=0.001,rho=2,tau=15,sigma=0.25,x0=np.array([1,1])):

	allPaths=[]
	t = np.arange(0,T,dt)
	#print("not adding noise")
	
	def f(x,t):
		dx1 = rho * tau * (x[0] - x[0]**3/3 - x[1])
		dx2 = tau/rho * x[0]
		return np.hstack([dx1,dx2])
	def g(x,t):
		return sigma*x
	
	def dW(dt):
		return np.random.normal(loc=np.zeros((2,)),scale=np.sqrt(dt))
	
	for ii in range(n):

		xnot = x0 + 0.03**2 * np.random.randn(2)
		xx = [xnot]

		for jj in range(1,len(t)+1):

			x = xx[jj-1]
			tt = t[jj-1]
			xx.append(x + f(x,tt)*dt + g(x,tt)*dW(dt))

		xx = np.vstack(xx)
		assert xx.shape[0] == (len(t) + 1), print(xx.shape)
		
		allPaths.append(xx)

	return allPaths

def generate_2d_swirls(n=100,T=1,dt=0.001,
					   omegas=(np.pi/5,-np.pi/2.5),bounds=((0.5,1),(1.25,1.5)),sigma=0.,
					   seed=1040):
	"""
	Makes the circles dataset (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html#sklearn.datasets.make_circles),
	but as a dynamical system. 
	Takes as arguments:
		n: number of trajectories to make
		T: integration time
		dt: integration timestep
		omegas: rotation speed for (inner,outer) circles
		sigma: sd of noise added to rotation (default zero)
		bounds: define width of each circle. arranged as:
				((inner circle inner bound, inner circle outer bound),
				 (outer circle inner bound, outer circle outer bound))

	Returns:
		trajectories:
			list of n np.ndarrays of size (T/dt)x2, each element corresponding to a trajectory
	"""    

	trajectories=[]
	t = np.arange(0,T,dt)
	gen = np.random.default_rng(seed=seed)
	
	def f(theta,t,omega):
		dtheta = omega*dt
		return dtheta
		
	def g(x,t):
		return sigma*x
	
	def dW(dt):
		return gen.multivariate_normal(mean=np.zeros((2,)),cov=dt*np.eye(2))

	
	for ii in range(n):
		
		circle = gen.binomial(1,0.5)
		r = gen.uniform(bounds[circle][0],bounds[circle][1])
		theta = gen.uniform(0,2*np.pi)
		omega = omegas[circle]

		xnot = np.hstack([r*np.cos(theta), r*np.sin(theta)])
		xx = [xnot]
		
		for jj in range(1,len(t)+1):
			x = xx[jj-1]
			tt = t[jj-1]
			dtheta = f(theta,tt,omega)
			theta += dtheta
			xx2 = np.hstack([r*np.cos(theta), r*np.sin(theta)])
			xx.append(xx2 + g(x,tt)*dW(dt))
		xx = np.vstack(xx)
		
		trajectories.append(xx)
	
	
	return trajectories

def generate_radial_odes(n=100,T=1,dt=0.001,
                        coeffs=[1.5,2,np.pi/4],sigma=0.,
                        seed=1040):
    """
    Makes the circles dataset (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html#sklearn.datasets.make_circles),
    but as a dynamical system, but with a double well potential on the radius 
    Takes as arguments:
    n: number of trajectories to make
    T: integration time
    dt: integration timestep
    coeffs:
        center of double well, weight on quadratic term, rotation per second, weight on dR

    Returns:
    trajectories:
    list of n np.ndarrays of size (T/dt)x2, each element corresponding to a trajectory
    """    

    trajectories=[]
    t = np.arange(0,T,dt)
    gen = np.random.default_rng(seed=seed)
    
    r0,a,omega = coeffs

    def ft(theta,t,omega):
        #dtheta = omega
        return omega
    
    def fr(r,t):
        
        # potential function: (r - r0)^4 - a(r - r0)^2
        return -(4 * (r - r0)**3 - 2*a * (r - r0))

    def g(x,t):
        return sigma*np.eye(2)

    def dW(dt):
        return gen.multivariate_normal(mean=np.zeros((2,)),cov = dt*np.eye(2))


    for ii in range(n):

        xnot = gen.multivariate_normal(mean=np.zeros((2,)),cov = a*np.eye(2))
     
        theta = np.arctan2(xnot[1],xnot[0])
        r = np.linalg.norm(xnot)
        xx = [xnot]
        for jj in range(1,len(t)+1):
            x = xx[jj-1]
            tt = t[jj-1]
            if r < r0:
                dtheta = ft(theta,tt,omega)
            else:
                dtheta = ft(theta,tt,-omega)
            dr = fr(r,tt)
            theta += dtheta*dt
            r += dr*dt
            r = max(0,r)
            xy = np.array([r*np.cos(theta),r*np.sin(theta)])
            dw_xy = g(xy,tt) @ dW(dt)
            xy2 = xy + dw_xy

            theta = np.arctan2(xy2[1],xy2[0])
            r = np.linalg.norm(xy2)
            xx2 = np.hstack(xy2)
            xx.append(xx2)
        xx = np.vstack(xx)

        trajectories.append(xx)


    return trajectories

def generate_stochastic_lorenz63(n=100,T=1,dt=0.001,coeffs=[10,28,8/3,0.,0.,0.],seed=1024):

	"""
	Makes a stochastic lorenz attractor 
	Takes as arguments:
		n: number of trajectories to make
		T: integration time
		dt: integration timestep
		coeffs:
			[sigma,rho,beta,sd1,sd2,sd3]
		where sd1-3 are dimension-wise sds of added noise
		sigma: sd of noise added to rotation (default zero)
		seed: seed for rng
	
	Returns:
		trajectories:
			list of n np.ndarrays of size (T/dt)x2, each element corresponding to a trajectory
	"""    
	
	sigma,rho,beta = coeffs[0],coeffs[1],coeffs[2]
	A = np.array([coeffs[3],coeffs[4],coeffs[5]])
	Sig = np.eye(3) * A
	t = np.arange(0,T,dt)
	gen = np.random.default_rng(seed=seed)
	
	allPaths = [ ]
	
	def f(x,t):
		dx = sigma * (x[1] - x[0]) #+ sample_dW[0]
		dy = (x[0] * (rho - x[2]) - x[1]) #+ sample_dW[1]
		dz = (x[0]*x[1]  - beta*x[2]) #+ sample_dW[2]
		return np.hstack([dx,dy,dz])
	
	def g(x,t):
		return Sig
	
	def dW(dt):
		return gen.multivariate_normal(mean=np.zeros((3,)),cov=np.eye(3)*dt)
	
	for ii in range(n):
	
		
		xnot = gen.multivariate_normal(mean=np.zeros((3,)),cov=np.eye(3))
		
		
		xx = [xnot]
		for jj in range(1,len(t)+1):
			
			x = xx[jj-1]
			tt = t[jj-1]
			xx.append(x + f(x,tt)*dt + g(x,tt)@dW(dt))
		
		xx = np.vstack(xx)
		assert xx.shape[0] == (len(t) + 1), print(xx.shape)
		
		allPaths.append(xx)
	
	
	return allPaths

def generate_stochastic_lorenz96(n=100,T=1,d=10,dt=0.001,coeffs=[8,0],seed=1024):

	"""
	Makes a stochastic high-d lorenz attractor 
	Takes as arguments:
		n: number of trajectories to make
		T: integration time
		dt: integration timestep
		coeffs:
			[sigma,rho,beta,sd1,sd2,sd3]
		where sd1-3 are dimension-wise sds of added noise
		sigma: sd of noise added to rotation (default zero)
		seed: seed for rng
	
	Returns:
		trajectories:
			list of n np.ndarrays of size (T/dt)x2, each element corresponding to a trajectory
	"""    
	
	F,sigma = coeffs[0],coeffs[1]
	t = np.arange(0,T,dt)
	gen = np.random.default_rng(seed=seed)
	
	allPaths = [ ]
	
	def f(x,t):
		dx = np.zeros(d)
		# Loops over indices (with operations and Python underflow indexing handling edge cases)
		for i in range(d):
			dx[i] = (x[(i + 1) % d] - x[i - 2]) * x[i - 1] - x[i] + F
		return dx
	
	def g(x,t):
		return np.eye(d)*sigma
	
	def dW(dt):
		return gen.multivariate_normal(mean=np.zeros((d,)),cov=np.eye(d)*dt)
	
	for ii in range(n):
	
		
		xnot = gen.multivariate_normal(mean=np.zeros((d,)),cov=np.eye(d))
		
		
		xx = [xnot]
		for jj in range(1,len(t)+1):
			
			x = xx[jj-1]
			tt = t[jj-1]
			xx.append(x + f(x,tt)*dt + g(x,tt)@dW(dt))
		
		xx = np.vstack(xx)
		assert xx.shape[0] == (len(t) + 1), print(xx.shape)
		
		allPaths.append(xx)
	
	
	return allPaths

def generate_stochastic_rossler(n=100,T=1,dt=0.001,coeffs=[0.1,0.1,14,0.,0.,0.],seed=1024):

	"""
	Makes a stochastic rossler attractor 
	Takes as arguments:
		n: number of trajectories to make
		T: integration time
		dt: integration timestep
		coeffs:
			[a,b,c,sd1,sd2,sd3]
		where sd1-3 are dimension-wise sds of added noise
		sigma: sd of noise added to rotation (default zero)
		seed: seed for rng
	
	Returns:
		trajectories:
			list of n np.ndarrays of size (T/dt)x2, each element corresponding to a trajectory
	"""    
	
	a,b,c = coeffs[0],coeffs[1],coeffs[2]
	A = np.array([coeffs[3],coeffs[4],coeffs[5]])
	Sig = np.eye(3) * A
	t = np.arange(0,T,dt)
	gen = np.random.default_rng(seed=seed)
	
	allPaths = [ ]
	
	def f(x,t):
		dx = -x[1] - x[2] #+ sample_dW[0]
		dy = x[0] + a*x[1] #+ sample_dW[1]
		dz = b + x[2]*(x[0] - c) #+ sample_dW[2]
		return np.hstack([dx,dy,dz])
	
	def g(x,t):
		return Sig
	
	def dW(dt):
		return gen.multivariate_normal(mean=np.zeros((3,)),cov=np.eye(3)*dt)
	
	for ii in range(n):
	
		
		xnot = gen.multivariate_normal(mean=np.zeros((3,)),cov=np.eye(3))
		
		
		xx = [xnot]
		
		for jj in range(1,len(t)+1):
			
			x = xx[jj-1]
			tt = t[jj-1]

			xx.append(x + f(x,tt)*dt + g(x,tt)@dW(dt))
		
		xx= np.vstack(xx)
		
		assert xx.shape[0] == len(t) +1, print(xx.shape)
		
		allPaths.append(xx)
	
	
	return allPaths
	
def generate_double_sde(n=100,T=1,dt=0.001,
						omegas=(np.pi/5,-np.pi/2.5),centers=((-0.5,0),(0.5,0)),sigma=0.,
						seed=1040):
	"""
	Uses the circle dynamics to create two opposing dynamical systems, separated at the x=0.
	Takes as arguments:
	n: number of trajectories to make
	T: integration time
	dt: integration timestep
	omegas: rotation speed for (left,right) circles
	sigma: sd of noise added to rotation (default zero)
	centers: define center of each circle:
	((left circle x, left circle y),
	(right circle x, right circle y))
	
	Returns:
	trajectories:
	list of n np.ndarrays of size (T/dt)x2, each element corresponding to a trajectory
	"""    

	trajectories=[]
	t = np.arange(0,T,dt)
	gen = np.random.default_rng(seed=seed)
	centers = (np.array(centers[0]),np.array(centers[1]))
	
	def f1(theta,t):
		dtheta = omegas[0]*dt
		return dtheta
	def f2(theta,t):
		dtheta = omegas[1]*dt
		return dtheta
		
	def g(x,t):
		return sigma * np.eye(2)#/(np.abs(x[0])+1/10)
	
	def dW(dt):
		return gen.normal(loc=np.zeros((2,)),scale=np.sqrt(dt))
	
	def polar_to_cartesian(r,theta):
	
		return np.hstack([r*np.cos(theta),r*np.sin(theta)])
	
	def cartesian_to_polar(xy):
	
		r = np.linalg.norm(xy)
		theta = np.arctan(xy[1]/xy[0]) + np.pi * (xy[0] < 0)
		return r,theta

	for ii in range(n):
		
		xnot = gen.normal(loc=[0,-0.5],scale=0.25)
		
		xx = [xnot]
		
		for jj in range(1,len(t)+1):
			x = xx[jj-1]
			tt = t[jj-1]

			if x[0] < 0:
				v = x - centers[0]
				r,theta = cartesian_to_polar(v)
				dtheta = f1(theta,tt)
				theta += dtheta
				xx2 = polar_to_cartesian(r,theta) + centers[0]
			else:
				v = x - centers[1]
				r,theta = cartesian_to_polar(v)
				dtheta = f2(theta,tt)
				theta += dtheta
				
				xx2 = polar_to_cartesian(r,theta) + centers[1]
			#print(xx2.shape)
			#g=g(x,tt)
			#dw = dW(dt)
			#print(g.shape)
			#print(dw.shape)
			#print((g@dw).shape)
   
			xx.append(xx2 + g(x,tt)@dW(dt))
		xx = np.vstack(xx)
		
		trajectories.append(xx)
	
	
	return trajectories

def generate_OU_balls(n=100,T=1,dt=0.001,imshape=(64,64),radius=2,
						coeffs =np.array([-4,-4]),
						sigma=0., 
						seed=1040):
	

	trajectories=[]
	t = np.arange(0,T,dt)
	gen = np.random.default_rng(seed=seed)
	centers = (np.array(centers[0]),np.array(centers[1]))

	def f(x,t):

		return coeffs @ x 
	
	def g(x,t):

		return sigma*np.eye(2)
	
	def dW(dt):

		return gen.multivariate_normal(mean=np.zeros((2,)),cov=dt*np.eye(2))
	

	for ii in range(n):
	
		xnot = gen.normal(loc=[0,0],scale=0.25)
		
		xx = [xnot]
		
		for jj in range(1,len(t)+1):

			x = xx[jj-1]
			tt = t[jj-1]

			xx.append(x + f(x,tt) + g(x,tt)@dW(dt))

		xx = np.vstack(xx)
		xx -= np.amin(xx)
		xx /= np.amax(xx)

		trajectories.append(xx)

	movies = traj_to_movie(trajectories,imshape,radius)
	
	return trajectories,movies

def traj_to_movie(trajectories,image_shape,radius):

	"""
	converts a latent trajectory to a movie
	"""

	movies = []
	gX,gY = np.meshgrid(np.linaspace(-1,1,16),np.linspace(-1,1,16))
	ball = (gX**2 + gY**2 < 1)

	for traj in trajectories:
		frames = np.zeros((len(traj),1,image_shape[0],image_shape[1]))

		for point in range(len(traj)):

			ballBound = (traj[point,:] *(image_shape - radius)).astype(int)
			frames[point,0,ballBound[0]:ballBound[0] + ball.shape[0], ballBound[1]:ballBound[1] + ball.shape[1]] += ball 

		movies.append(frames)
	return movies
	

def downsample(data:list,origdt:float,newdt:float,noise:bool=True) -> np.ndarray:


	skip = int(newdt/origdt)

	downsampled = [d[::skip] for d in data]

	if noise:
		downsampled = [d + 0.01*np.random.randn(*d.shape) for d in downsampled]

	return downsampled



class toyDataset(Dataset):

	"""
	dataloader for toy datasets. expects data in the form of that created by
	my toy data creation methods -- in other words, a list of np.arrays
	flattens all arrays and creates a set of valid indices of that array to sample from.
	This set of valid indices is also based on nForward: the number of steps forward in time
	that we want our model to predict
	"""

	def __init__(self,data,dt,nForward=1) -> None:
		
		

		self.maxForward = nForward
		exampleInd = np.random.choice(len(data),1)[0]
		self.exampleTraj = data[exampleInd]
		lens = list(map(len,data))
		lens2 = [0] + list(np.cumsum([l for l in lens][:-1]))
		sets = [np.vstack([np.arange(ii, l+ ii - self.maxForward) for ii in range(self.maxForward + 1)]).T for l in lens]
		#pairs = [np.vstack([np.arange(0,l-1),np.arange(1,l)]).T for l in lens]
		sumSets = [p+l for p,l in zip(sets,lens2)]
		validInds = np.vstack(sumSets)
		self.data= np.vstack(data)
		self.data_inds = validInds
		self.dt = dt
		self.length = len(validInds)
		#print('added in more forward predictions')
		## needed: slice data by dt? need true dt, ds dt for that
		## should be fine to add though

	def __len__(self):

		return self.length 
	
	def __getitem__(self, index):
		
		single_index = False
		result = []
		try:
			iterator = iter(index)
		except TypeError:
			index = [index]
			single_index = True

		for ii in index:
			inds = self.data_inds[ii]

			samples = [self.transform(self.data[ind]) for ind in inds]
			samples.append(self.dt)			
			#s1,s2 = self.transform(self.data[inds[0]]),self.transform(self.data[inds[1]])
			result.append(samples)

		if single_index:
			return result[0]
		return result
	
	def transform(self,data):
		return torch.from_numpy(data).type(torch.FloatTensor)

