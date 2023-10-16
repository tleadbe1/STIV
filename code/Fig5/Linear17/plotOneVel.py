import jax.numpy as np
import matplotlib.pyplot as plt
from jax.scipy.special import erf
from jax import vmap

cdf = lambda x: 1/2*(1 + erf(x/np.sqrt(2)))
pdf = lambda x: 1/np.sqrt(2*np.pi) * np.exp(-x**2 / 2)

from inputs import *

mus = np.load("musLong.npy")
Ss = np.load("SsLong.npy")
tsE = np.load("tsLong.npy")

xs = np.load("xs.npy")
tsL = np.load("ts.npy")

ellE = ell(tsE)
ellL = ell(tsL)

lEP = np.load("langevinEP.npy")
lFex = np.load("langevinFex.npy")

eEP = np.load("eyinkEPLong.npy")
eFex = np.load("eyinkFexLong.npy")


augxs = np.zeros((xs.shape[0],xs.shape[1],xs.shape[2] +2))
augxs = augxs.at[:,:,1:-1].set(xs)
augxs = augxs.at[:,:,-1].set((np.ones((xs.shape[1],xs.shape[0])) * ell(tsL)).T)
dx = augxs[:,:,1:] - augxs[:,:,:-1]
phiL = np.mean(dx > 0.,axis = 1)


augmus = np.zeros((mus.shape[0],mus.shape[1]+2))
augmus = augmus.at[:,1:-1].set( mus)
augmus = augmus.at[:,-1].set(ell(tsE))
dmus = augmus[:,1:] - augmus[:,:-1]

D1 = np.diag(np.ones(Ss.shape[-1])) - np.diag(np.ones(Ss.shape[-1]-1),-1)

def dif(M):
  return D1 @ M @ D1.T


ds = np.zeros((Ss.shape[0],Ss.shape[-1] + 1))
ds = ds.at[:,:-1].set(np.sqrt(vmap(np.diag)(vmap(dif)(vmap(np.linalg.inv)(Ss)))))
ds = ds.at[:,-1].set(np.sqrt(vmap(np.linalg.inv)(Ss)[:,-1,-1]))

phiE = cdf(dmus/ds)




fig,axes = plt.subplots(2,2,figsize = (16,9))

mxs = np.mean(augxs,axis = 1)


for i in range(mxs.shape[-1]):
  axes[0,0].plot(tsL,mxs[:,i],color = "blue",linestyle = "dashed",zorder = 1)
  axes[0,0].plot(tsE,augmus[:,i],color = "black",linestyle = "solid",zorder = 0)
  axes[1,0].plot(tsL,phiL[:,i],color = "blue",linestyle = "dashed",zorder = 1)
  axes[1,0].plot(tsE,phiE[:,i],color = "black",linestyle = "solid",zorder = 0)




axes[0,1].plot(tsL,lFex,color = "blue",linestyle = "dashed",zorder = 1)
axes[0,1].plot(tsE,eFex,color = "black",linestyle = "solid",zorder = 0)
axes[1,1].plot(tsL,lEP,color = "blue",linestyle = "dashed",zorder = 1)
axes[1,1].plot(tsE,eEP,color = "black",linestyle = "solid",zorder = 0)



fig.savefig("SingleVel.pdf")

