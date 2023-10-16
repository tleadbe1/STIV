import jax.numpy as np
from jax import random
import sys
sys.path.append("/code/DynamicsSolvers")
from eyinkSigmaInvDW import *
from langevinDW import *
from inputs import *

x1 = ell(0)
ell = lambda t: x1

nSim = 100000 # 1e5

eSim = EyinkSim(k1 = k1,l1 = l1,k2 = k2,l2 = l2,beta = beta,eta = eta,ell = ell,nMasses = nMasses)
lSim = LangevinDynamics(l1=l1,k1=k1,k2 = k2,l2=l2,eta = eta,beta = beta,ell = ell,nMasses = nMasses)


x = -lSim.l1*np.arange(1,nMasses + 1)*np.ones((nSim,nMasses))

t0 = 0.
W0 = np.zeros(nSim)
dt = float(1e-4)
nSteps = int(3/dt)

key,x,_,_ = lSim.run(key,x,t0,W0,dt,nSteps)


np.save("x0.npy",x)

