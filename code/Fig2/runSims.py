import jax.numpy as np
from jax import random
import sys
sys.path.append("/code/DynamicsSolvers")
from eyinkSP import *
from langevinSP import *

from inputs import *
from jax.config import config


eSim = EyinkSim(k = k,beta = beta,l = l,eta = eta,ell = ell,nMasses = nMasses)
lSim = LangevinDynamics(l=l,k=k,eta = eta,beta = beta,ell = ell,nMasses = nMasses)

key,k0 = random.split(key)
nSim = int(1e5)

mu = 0.
s = np.sqrt(2/eSim.k/eSim.beta)

x0 = random.normal(k0,(nSim,))*s + mu







W0 = np.zeros((len(x0),))
dt = 0.01
t0 = 0.
t = t0
x = x0
W = W0
xs = [x,]
mus = [mu,]
sigs = [s,]
ts = [t,]
Ws = [W,]
for i in range(200):
  key,x,_,W = lSim.run(key,x,t,W,dt/100,500)
  xs.append(x)
  Ws.append(W)
  mu,s,t = eSim.run(mu,s,t,dt,5)
  mus.append(mu)
  sigs.append(s)
  ts.append(t)

xs = np.array(xs)
ts = np.array(ts)
Ws = np.array(Ws)
mus = np.array(mus)
sigs = np.array(sigs)

np.save("xs.npy",np.array(xs))
np.save("Ws.npy",np.array(Ws))
np.save("mus.npy",np.array(mus))
np.save("sigs.npy",np.array(sigs))
np.save("ts.npy",np.array(ts))
