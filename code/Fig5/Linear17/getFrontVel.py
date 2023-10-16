import numpy as np

import jax.numpy as np
from jax import jit,vmap,random
from jax.scipy.special import erf
import matplotlib.pyplot as plt
import sys
sys.path.append("/code/DynamicsSolvers")
from eyinkSigmaInvDW import *
from langevinDW import *
from basic_inputs import *
import os
from bisect import bisect_left as bl
from scipy.stats import linregress as lr




def cdf(x):
  return 1/2*(1 + erf(x/np.sqrt(2)))

def pdf(x):
  return np.exp(-x**2/2)/np.sqrt(2*np.pi)


def getVTimes(ts,phases):
  t = []
  for p in phases.T:
    i = bl(p,.5)
    t.append(1/2*(ts[i-1] + ts[i]))
  return np.array(t)

def phases(emu,S):
  dmu = emu[1:] - emu[:-1]
  n = S.shape[0]
  D1 = np.diag(np.ones(n)) - np.diag(np.ones(n-1),-1)
  s = np.zeros(n+1)
  s = s.at[:-1].set(np.sqrt(np.diag(D1 @ S @ D1.T)))
  s = s.at[-1].set(np.sqrt(S[-1,-1]))
  return cdf(dmu/s)



def getFrontTimes(mus,Ss,xs,lts,ets,u):
  
  mxs = np.mean(xs,axis = 1)
  
  N = mus.shape[-1]
  
  lls = -l10*(1. -  u*lts)
  els = -l10*(1. -  u*ets)
  
  exs = np.zeros((xs.shape[0],xs.shape[1],mxs.shape[1]+2))
  exs = exs.at[:,:,1:-1].set(xs)
  exs = exs.at[:,:,-1].set((np.ones((xs.shape[1],xs.shape[0]))*lls).T)
  
  dxs = exs[:,:,1:] - exs[:,:,:-1] 
  lPhases = np.mean(dxs > 0.,axis = 1)
  
  emus = np.zeros((mus.shape[0],mus.shape[1]+2))
  emus = emus.at[:,1:-1].set(mus)
  emus = emus.at[:,-1].set(els)
  

  
  ePhases = vmap(phases)(emus,Ss)
  
  eTs = getVTimes(ets,ePhases)
  lTs = getVTimes(lts,lPhases)  
  
  us = np.linspace(0,1,N+2)[1:]

  velE = lr(eTs,us).slope
  velL = lr(lTs,us).slope

  return (velL,velE),lTs,eTs,us,#np.array(velAprx)

# ------------------------


files = ["V1_16","V1_8","V1_4","V1_2","V1","V2","V4","V6","V8","V10"]
shearRates = [1/16,1/8,1/4,1/2,1.,2.,4.,6.,8.,10.]

Lvs = []
Evs = []
eTimes = []
lTimes = []
i = 0
for f,u in zip(files,shearRates):
  os.chdir("/code/Fig5/Linear17/" + f  + "/")
  mus = np.load("musLong.npy")
  Ss = np.load("SsLong.npy")
  tsLong = np.load("tsLong.npy")
  ts = np.load("ts.npy")
  xs = np.load("xs.npy")
  (vL,vE),lTs,eTs,us = getFrontTimes(mus,Ss,xs,ts,tsLong,u) 
  Lvs.append(vL)
  Evs.append(vE)
  eTimes.append(eTs)
  lTimes.append(lTs)
  os.chdir("/code/Fig5/Linear17/")
  i += 1

os.chdir("/code/Fig5/Linear17/VelData/")
np.save("langVels.npy",np.array(Lvs))
np.save("eynVels.npy",np.array(Evs))
np.save("langTimes.npy",np.array(lTimes))
np.save("eynTimes.npy",np.array(eTimes))
np.save("shearRates.npy",np.array(shearRates))
os.chdir("/code/Fig5/Linear17/")


