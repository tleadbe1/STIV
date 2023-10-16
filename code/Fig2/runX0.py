import jax.numpy as np
from jax import random

import sys
sys.path.append("/code/DynamicsSolvers")
from eyinkSP import *
from langevinSP import *


from inputs import *
from jax.config import config


ell0 = ell(0)
ell = lambda t: ell0


eSim = EyinkSim(k = k,beta = beta,l = l,eta = eta,ell = ell,nMasses = nMasses)
lSim = LangevinDynamics(l=l,k=k,eta = eta,beta = beta,ell = ell,nMasses = nMasses)

nSim = 100000 # 100000

key,k0 = random.split(key)

x = random.normal(k0,(nSim,))*np.sqrt(1/k/beta)

W = np.zeros((nSim,))
dt = 0.01
t0 = 0.
t = t0
key,x,_,_ = lSim.run(key,x,t,W,dt/100,5000)


np.save("x0.npy",np.array(x))
