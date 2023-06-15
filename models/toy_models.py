import torch
import numpy as np
import matplotlib.pyplot as plt


def generate_ndim_benes(n=100,d = 20,T=100,dt=1):

    t = np.arange(0,T,dt)
    
    allPaths = []

    for ii in range(n):

        sample_dW = dt * np.random.randn((len(t),d))
        xnot = np.random.randn(d)

        xx = [xnot]
        for jj in range(len(t)):

            dx = np.tanh(xx[jj])*dt + sample_dW[jj]
            xx.append(xx[jj] + dx)

        xx = np.vstack(xx)
        assert xx.shape[0] == (len(t) + 1), print(xx.shape)
        allPaths.append(xx)

    return allPaths


def plot_sde(data):

    ax = plt.gca()
    for traj in data:

        ax.plot(traj[:,0],traj[:,1])
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")

    plt.savefig()


if __name__ == '__main__':

    xs = generate_ndim_benes(100,d=3)
