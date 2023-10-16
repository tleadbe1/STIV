from jax import config
config.update("jax_enable_x64",True)

import jax.numpy as np
from jax import random
import sys
sys.path.append("/code/DynamicsSolvers")
from eyinkSigmaInvDW import *
from langevinDW import *
from inputs import *
import gc


runEyink = True
runLangevin = False
join = True


eSim = EyinkSim(k1 = k1,l1 = l1,k2 = k2,l2 = l2,beta = beta,eta = eta,ell = ell,nMasses = nMasses)
lSim = LangevinDynamics(l1=l1,k1=k1,k2 = k2,l2=l2,eta = eta,beta = beta,ell = ell,nMasses = nMasses)


xs0 = np.load("x0.npy")
mu0 = np.mean(xs0,axis = 0)
S0 = np.linalg.inv(np.cov(xs0.T))




W0 = np.zeros((xs0.shape[0],))
t0 = 0.
t = t0
x = xs0
W = W0
if runLangevin:
  print("Running Langevin Sims")
  _,xs,ts,Ws = lSim.runCollect(key,x,t,W,dtL,nEval_l,nStepL)

  np.save("xs.npy",np.array(xs))
  np.save("Ws.npy",np.array(Ws))
  np.save("ts.npy",np.array(ts))



t = 0.
mu = mu0
S = S0
if runEyink:
  print("Running Eyink Solver")
  for j in range(nRep):
    mus,Ss,ts = eSim.runCollect(mu,S,t,dtE,nEval_e//nRep,nStepE)
    mu = mus[-1]
    S = Ss[-1]
    t = ts[-1]
    np.save("mus%d.npy"%(j),np.array(mus))
    np.save("Ss%d.npy" %(j),np.array(Ss))
    np.save("ts%d.npy" %(j),np.array(ts))
    del ts
    del mus
    del Ss 
    gc.collect()
    print((j+1)/nRep)


if join:
  import os
  mus = np.load("mus0.npy")
  Ss = np.load("Ss0.npy")
  ts = np.load("ts0.npy")
  os.remove("mus0.npy") 
  os.remove("Ss0.npy") 
  os.remove("ts0.npy") 
  for j in range(1,nRep):
    mus = np.concatenate((mus,np.load("mus%d.npy"%(j))[1:]),axis = 0)
    Ss = np.concatenate((Ss,np.load("Ss%d.npy"%(j))[1:]),axis = 0)
    ts = np.concatenate((ts,np.load("ts%d.npy"%(j))[1:]),axis = 0)
    os.remove("mus%d.npy"%(j))
    os.remove("Ss%d.npy"%(j))
    os.remove("ts%d.npy"%(j))
  np.save("musLong.npy",mus)
  np.save("SsLong.npy",Ss)
  np.save("tsLong.npy",ts)



