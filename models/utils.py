import numpy as np


def rbf_dot(patterns1,patterns2,sig):

    size1 = patterns1.shape
    size2 = patterns2.shape

    G = np.pow(patterns1,2).sum(axis=1)
    H = np.pow(patterns2,2).sum(axis=1)

    Q = np.tile(G,[1,size2[0]])
    R = np.tile(H.T,[size1[0],1])

    H = Q + R - 2*patterns1 @ patterns2.T

    H=np.exp(-H/2/sig**2);

    return H 

def mmd_fxn(lat1,lat2,sig):

    if sig == -1:
        Z = np.hstack([lat1,lat2])

        size1 = Z.shape[0]
        if size1 > 100:
            Zmed = Z[0:100,:]
            size1 = 100
        else:
            Zmed = Z 

        G = np.sum(Zmed **2,axis=1)
        Q = np.tile(G,[1,size1])
        R = np.tile(G.T,[size1,1])

        dists = Q + R - 2* Zmed @ Zmed.T
        dists = dists - np.tril(dists)
        dists = np.reshape(dists,[size1**2,1])
        sig = np.sqrt(0.5 * np.median(dists[dists > 0]))

    mmds = []
    for l in lat1:
        l = np.expand_dims(l,0)
        mmd = np.sum(rbf_dot(l,lat1) - rbf_dot(l,lat2))
        mmds.append(mmd)

    return mmds


if __name__ == '__main__':

    pass