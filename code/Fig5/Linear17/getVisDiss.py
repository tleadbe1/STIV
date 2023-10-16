import numpy as onp
import jax.numpy as np
from jax import jit,vmap,random
from jax.scipy.stats.norm import pdf,cdf
import sys
sys.path.append("/code/DynamicsSolvers")
from eyinkSigmaInvDW import *
from langevinDW import *
from basic_inputs import *
import os


    





files =  ["V1_16","V1_8","V1_4","V1_2","V1","V2","V4","V6","V8","V10"]
shearRates = [1/16,1/8,1/4,1/2,1.,2.,4.,6.,8.,10.]


lVisDiss = None
eVisDiss = None

i = 0
for f,u in zip(files,shearRates):
  ell = lambda t: -l10*(1 - u*t)
  os.chdir("/code/Fig5/Linear17/" + f  + "/")
  mus = np.load("musLong.npy")
  Ss = np.load("SsLong.npy")
  tsLong = np.load("tsLong.npy")
  ts = np.load("ts.npy")
  xs = np.load("xs.npy")
  mxs = np.mean(xs,axis = 1)

  Dc = np.diag(np.ones(mxs.shape[0]-1),1) - np.diag(np.ones(mxs.shape[0]-1),-1)
  Dc = Dc.at[0,0].set(-2)
  Dc = Dc.at[0,1].set(2)
  Dc = Dc.at[-1,-1].set(2)
  Dc = Dc.at[-1,-2].set(-2)
  Dc = Dc/(ts[2] - ts[0])
  sim = EyinkSim(k1,l1,k2,l2,eta,beta,lambda t: -l10*(1-u*t),nMasses)
  mxDots = Dc @ mxs
  muDots,_ = vmap(sim.alphaDot)(mus,Ss,tsLong)
  lVisDiss = eta* np.sum(mxDots**2,axis = -1)
  eVisDiss = eta*np.sum(muDots**2,axis = -1)
  os.chdir("/code/Fig5/Linear17/VisDissData/")
  np.save("lVisDiss%s.npy" %(f),lVisDiss)
  np.save("eVisDiss%s.npy" %(f),eVisDiss)
  os.chdir("/code/Fig5/Linear17/")
  i += 1





