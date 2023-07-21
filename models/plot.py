import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import matplotlib
def plotSamples3d(true,generated,n=100):

	fig = plt.figure()
	ax1 = fig.add_subplot(121,projection='3d')
	ax2 = fig.add_subplot(122,projection='3d')
	order = np.random.choice(len(true),100,replace=False)
	for o in order:
		t = true[o]
		ax1.plot(t[:,0],t[:,1],t[:,2])
	ax1.set_xticks([])
	ax1.set_yticks([])
	ax1.set_zticks([])
		
	order = np.random.choice(len(true),100,replace=False)

	for o in order:
		g = generated[o]
		ax2.plot(g[:,0],g[:,1],g[:,2])
	ax2.set_xticks([])
	ax2.set_yticks([])
	ax2.set_zticks([])
	plt.show()

def plotSamples1d(true,generated):
	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	order = np.random.choice(len(true),100,replace=False)
	true_lims = (np.amin(true)-0.1,np.amax(true)+0.1)
	for o in order:
		t = true[o].squeeze()
		time = list(range(1,len(t)+1))
		ax1.plot(time,t)
	ax1.set_xticks([])
	ax1.set_ylim(true_lims)
	#ax1.set_yticks([])

	
	order = np.random.choice(len(generated),100,replace=False)

	for o in order:
		g = generated[o].squeeze()
		time = list(range(1,len(g)+1))
		ax2.plot(time,g)
	ax2.set_xticks([])
	ax2.set_ylim(true_lims)
	#ax1.set_yticks([])
	plt.show()


def sde_gif(data):

	fig = plt.figure()
	if data[0].shape[-1] == 2:
		ax = fig.add_subplot(111)
		for traj in data:

			ax.plot(traj[:,0],traj[:,1])
			ax.set_xlabel("dim 1")
			ax.set_ylabel("dim 2")

		plt.savefig("testingonetwo.png")
	else:
		
		for ii,traj in enumerate(data):
			mins = np.amin(traj,axis=0)
			maxs = np.amax(traj,axis=0)
			ax = fig.add_subplot(111,projection='3d')

			ax.set_xlim([mins[0],maxs[0]])
			ax.set_ylim([mins[1],maxs[1]])
			ax.set_zlim([mins[2],maxs[2]])

			line, = ax.plot([],[])
			ax.view_init(azim=180)

			def makeAni1(frame):
				if frame < traj.shape[0]:
					line.set_data_3d(traj[:frame+1,0],traj[:frame+1,1],traj[:frame+1,2])
				ax.view_init(azim=180 + frame/6)
				ax.set_xlabel("dim 1")
				ax.set_ylabel("dim 2")
				ax.set_zlabel("dim 3")
				return line,
		
			print("animating 1")
			anim= animation.FuncAnimation(fig,makeAni1,frames=3000,interval=20,blit=True)
			anim.save(f'changeAcrossTime{ii}.gif', writer = 'ffmpeg', fps = 50)
			plt.close('all')
			


		#plt.savefig("testingonetwothree.png")

def plot1dFlows(data,estimatedMus,truMu=1,dt=0.01):
	
	Vs = np.ones(estimatedMus.shape)*dt
	#print(len(data))
	#print(dt)
	Xs = np.arange(0,(len(data))*dt,dt)
	truMus = data * truMu *dt

	sqErr = (estimatedMus.squeeze() - truMus.squeeze())**2
	norm = matplotlib.colors.Normalize(vmin=np.amin(sqErr),vmax=np.amax(sqErr))
	cm = matplotlib.cm.gist_heat
	sm = matplotlib.cm.ScalarMappable(cmap='gist_heat',norm=norm)
	sm.set_array([])
	ax = plt.gca()
	#print(estimatedMus.shape)
	ax.quiver(Xs,data,Vs,truMus,color='k',angles='xy',label='true mus')
	ax.quiver(Xs,data,Vs,estimatedMus,sqErr,cmap='gist_heat',angles='xy',label='estimations',alpha=0.25)
	plt.colorbar(sm)
	ax.set_xlabel("Time (s)")
	ax.set_ylabel("X")
	plt.show()
	pass

def plot1dSigmas(data,estimatedSigmas,truSigma=0.5,dt=0.01):

	truSDs = truSigma*data*np.sqrt(dt) 
	sqErr = (estimatedSigmas.squeeze() - truSDs.squeeze())**2
	norm = matplotlib.colors.Normalize(vmin=np.amin(sqErr),vmax=np.amax(sqErr))
	cm = matplotlib.cm.gist_heat
	sm = matplotlib.cm.ScalarMappable(cmap='gist_heat',norm=norm)
	sm.set_array([])
	Xs = np.arange(0,(len(data))*dt,dt)
	ax = plt.gca()
	ax.errorbar(Xs,data,xerr=truSDs.squeeze(),color='k')
	ax.errorbar(Xs,data,xerr=estimatedSigmas.squeeze(),ecolor=cm(norm(sqErr)))
	ax.set_xlabel("Time (s)")
	ax.set_ylabel("X")
	plt.colorbar(sm)
	plt.show()


	
	pass