import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from my_models import linearLatentSDE,nonlinearLatentSDE
from train import train
from data import *
from tqdm import tqdm
import seaborn as sns

def generate_ndim_benes(n=100,d = 20,T=100,dt=1):

    t = np.arange(0,T,dt)
    
    allPaths = []

    for ii in range(n):

        sample_dW = dt * np.random.randn(len(t),d)
        xnot = np.zeros((d,))#0.01*np.random.randn(d)

        xx = [xnot]
        for jj in range(len(t)):

            dx = np.tanh(xx[jj])*dt + sample_dW[jj]

            xx.append(xx[jj] + dx)

        xx = np.vstack(xx)
        assert xx.shape[0] == (len(t) + 1), print(xx.shape)
        allPaths.append(xx)

    return allPaths

def generate_geometric_brownian(n=100,T=100,dt=1,mu=1,sigma=0.5,x0=0.1):

    allPaths=[]
    t = np.arange(0,T,dt)
    for ii in range(n):

        xnot = x0 + 0.03**2 * np.random.randn(1)
        xx = [xnot]

        for jj in range(1,len(t)+1):
            x = xx[jj-1]
            sample_dW = np.sqrt(dt) * np.random.randn(1)
            dx = mu*x *dt + sigma*x * sample_dW
            xx.append(xx[jj-1] + dx)
        xx = np.hstack(xx)[:,None]

        xx = xx + 0.01*np.random.randn(*xx.shape)
        allPaths.append(xx)

    return allPaths

def generate_stochastic_lorenz(n=100,T=100,dt=1,coeffs=[10,28,8/3,0.15,0.15,0.15]):

    sigma,rho,beta = coeffs[0],coeffs[1],coeffs[2]
    A1,A2,A3 = coeffs[0],coeffs[1],coeffs[2]
    t = np.arange(0,T,dt)
    assert len(t) == 40*5
    allPaths = [ ]

    for ii in range(n):

        
        xnot = np.random.randn(3)

        xx = [xnot]
        for jj in range(1,len(t)+1):

            prev = xx[jj-1]
            x = prev[0]
            y = prev[1]
            z = prev[2]

            sample_dW = np.sqrt(dt) * np.random.randn(3)
            dx = sigma * (y - x)*dt +A1 * sample_dW[0]
            dy = (x * (rho - z) - y)*dt +A2 * sample_dW[1]
            dz = (x*y  - beta*z)*dt +A3 * sample_dW[2]

            #assert (abs(dx) < 200) & (abs(dy) < 200) & (abs(dz)<200),print(dx,dy,dz,ii,jj)
            xx.append(xx[jj-1] + np.hstack([dx,dy,dz]))

        xx = np.vstack(xx)
        assert xx.shape[0] == (len(t) + 1), print(xx.shape)
        
        allPaths.append(xx)

    mu = np.nanmean(np.vstack(allPaths),axis=0)
    sd = np.nanstd(np.vstack(allPaths),axis=0)
    allPaths = [(p - mu[None,:])/sd[None,:] for p in allPaths]
    allPaths = [p + 0.01*np.random.randn(*p.shape) for p in allPaths]
    #allPaths = [p[::5] for p in allPaths]
    return allPaths

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
        time = list(range(1,len(t)+1))
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

def plotFlows2dLorenz(params,model):

    pass

if __name__ == '__main__':


    dt = 0.001
    xs = generate_geometric_brownian(1024,dt=0.01,T = 1)
    linearmodel = nonlinearLatentSDE(dim=1,diag_covar=True,save_dir='/home/miles/test1_linear')
    dls = makeToyDataloaders(np.vstack(xs),np.vstack(xs),dt=0.01)
    #linearmodel.load('/home/miles/test1_linear/checkpoint_300.tar')
    linearmodel = train(linearmodel,dls,nEpochs=1000,save_freq=100,test_freq=25)
    print(f"generating new data! {1024} samples")
    samples = []
    muDist = []
    sigDist=[]
    for ii in tqdm(range(256), desc="generating trajectories"):
        x = xs[ii]
        x = torch.from_numpy(x).type(torch.FloatTensor)
        mus = linearmodel.MLP(x[:-1,:]).detach().cpu().numpy().squeeze()
        muDist.append(mus/x[:-1].detach().cpu().numpy().squeeze())
        Ds = linearmodel.D(x[:-1]).detach().cpu().numpy().squeeze()
        sigDist.append(Ds/x[:-1].detach().cpu().numpy().squeeze())
        #steps = mus*dt + Ds*np.sqrt(dt)*np.random.randn(*(x[:-1].squeeze().shape))
        #steps = steps.detach().cpu().numpy()
        #xGen = x[:-1].detach().cpu().numpy().squeeze() + steps
        
        #samples.append(np.hstack([x[0],xGen.squeeze()]))
        
        x0 = 0.1 + 0.03**2 * np.random.randn(1)
        xsTmp = linearmodel.generate(x0,T=1,dt=0.01)
        samples.append(xsTmp)
    
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
    ax1.set_title(f"{np.nanmean(muDist)}")
    ax2.set_title(f"{np.nanmean(sigDist)}")
    plt.show()
    plt.close(fig)

    print("done!")
    plotSamples1d(xs,samples)
   
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
