import jax.numpy as np
from jax import random
import sys
sys.path.append("/code/DynamicsSolvers")
from eyinkSigmaInvDW import *
from langevinDW import *
from inputs import *


eSim = EyinkSim(k1 = k1,l1 = l1,k2 = k2,l2 = l2,beta = beta,eta = eta,ell = ell,nMasses = nMasses)
lSim = LangevinDynamics(l1=l1,k1=k1,k2 = k2,l2=l2,eta = eta,beta = beta,ell = ell,nMasses = nMasses)


xs0 = np.load("x0.npy")
mu0 = np.mean(xs0,axis = 0)
S0 = np.linalg.inv(np.cov(xs0.T))




W0 = np.zeros(xs0.shape[0])
dt = 0.01
t0 = 0.

def sf(c,n):
  key,x,W,mu,S,t = c
  key,x,_,W = lSim.run(key,x,t,W,dt/100,500)
  mu,S,t = eSim.run(mu,S,t,dt,5)
  c = key,x,W,mu,S,t
  return c,c

_,out = scan(sf,(key,xs0,W0,mu0,S0,t0),None,length = 200)
_,xs,Ws,mus,Ss,ts = out


np.save("xs.npy",np.array(xs))
np.save("Ws.npy",np.array(Ws))
np.save("mus.npy",np.array(mus))
np.save("SInvs.npy",np.array(Ss))
np.save("ts.npy",np.array(ts))
