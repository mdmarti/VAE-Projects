import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

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

def generate_stochastic_lorenz(n=100,T=100,dt=1,coeffs=[0.25,1.2,1,0.5,0.3,1.2]):

    sigma,rho,beta = coeffs[0],coeffs[1],coeffs[2]
    A1,A2,A3 = coeffs[0],coeffs[1],coeffs[2]
    t = np.arange(0,T,dt)
    
    allPaths = []

    for ii in range(n):

        sample_dW = dt * np.random.randn(len(t),3)
        xnot = 0.001*np.random.randn(3)

        xx = [xnot]
        for jj in range(len(t)):

            prev = xx[jj]
            x = prev[0]
            y = prev[1]
            z = prev[2]

            dx = sigma * (y - x)*dt +A1 * sample_dW[jj,0]
            dy = x * (rho - z)*dt +A2 * sample_dW[jj,1]
            dz = (x*y  - beta*z)*dt +A3 * sample_dW[jj,2]

            xx.append(xx[jj] + np.hstack([dx,dy,dz]))

        xx = np.vstack(xx)
        assert xx.shape[0] == (len(t) + 1), print(xx.shape)
        allPaths.append(xx)

    return allPaths


def plot_sde(data):

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
            ax = fig.add_subplot(111,projection='3d')
            ax.set_xlim([-4,4.0])
            ax.set_ylim([-4,4.0])
            ax.set_zlim([-4,4.0])
            line, = ax.plot([],[])
            ax.view_init(azim=180)
            def makeAni1(frame):
                if frame < traj.shape[0]:
                    line.set_data_3d(traj[:frame+1,0],traj[:frame+1,1],traj[:frame+1,2])
                ax.view_init(azim=180 + frame/6)

                return line,
        
            print("animating 1")
            anim= animation.FuncAnimation(fig,makeAni1,frames=3000,interval=20,blit=True)
            anim.save(f'changeAcrossTime{ii}.gif', writer = 'ffmpeg', fps = 50)
            plt.close('all')
            ax.set_xlabel("dim 1")
            ax.set_ylabel("dim 2")
            ax.set_zlabel("dim 3")

        #plt.savefig("testingonetwothree.png")


if __name__ == '__main__':

    xs = generate_ndim_benes(20,d=2,dt=0.001,T = 2)
    plot_sde(xs)
    xs = generate_stochastic_lorenz(3,dt=0.01,T = 100,coeffs=[10,28,8/3,1.2,0,0])
    plot_sde(xs)
