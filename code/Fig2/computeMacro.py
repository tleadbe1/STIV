import jax.numpy as np
from jax import random
import sys
sys.path.append("/code/DynamicsSolvers")
from eyinkSP import *
from langevinSP import *

from inputs import *

eSim = EyinkSim(k = k,beta = beta,l = l,eta = eta,ell = ell,nMasses = nMasses)
lSim = LangevinDynamics(l=l,k=k,eta = eta,beta = beta,ell = ell,nMasses = nMasses)





xs= np.load("xs.npy")
Ws = np.load("Ws.npy")
mus = np.load("mus.npy")
sigs = np.load("sigs.npy")
ts = np.load("ts.npy")

ents = []
for x,mu,s,t in zip(xs,mus,sigs,ts):
  ents.append(eSim.entropyProduction(mu,s,t))

lents = lSim.entropyProduction(xs,ts)


np.save("stivEP.npy",np.array(ents))
np.save("langevinEP.npy",lents)

lFex = np.mean(vmap(vmap(lSim.Fex,in_axes = (0,None)))(xs,ts),axis = 1)
eFex = vmap(eSim.Fex)(mus,sigs,ts)

np.save("langevinFex.npy",lFex)
np.save("stivFex.npy",eFex)

