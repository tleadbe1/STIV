import jax.numpy as np
import numpy as onp
from jax import vmap
import sys
sys.path.append("/code/DynamicsSolvers")
from eyinkSigmaInvDW import *
from langevinDW import *
from inputs import *
  
eSim = EyinkSim(k1 = k1,l1 = l1,k2 = k2,l2 = l2,beta = beta,eta = eta,ell = ell,nMasses = nMasses)
lSim = LangevinDynamics(k1=k1,l1=l1,k2=k2,l2=l2,eta = eta,beta = beta,ell = ell,nMasses = nMasses)


xs= np.load("xs.npy")
Ws = np.load("Ws.npy")
lts = np.load("ts.npy")
mus = np.load("musLong.npy")
Ss = np.load("SsLong.npy")
ts = np.load("tsLong.npy")

computeEyink = True
computeLang = True

if computeEyink:

  ents = vmap(eSim.entropyProduction)(mus,Ss,ts)
  np.save("eyinkEPLong.npy",np.array(ents))
  eFex = vmap(eSim.Fex)(mus,Ss,ts)
  np.save("eyinkFexLong.npy",eFex)

if computeLang:
  n = int(xs.shape[1])
  nmle = int(n*0.02)
  nkde = int(n*0.49)
  temp1,temp2 = lSim.entropyProduction(xs[:,:nmle],xs[:,nmle:nmle+nkde],xs[:,nmle+nkde:],lts,Ws,0.2)
  lents = temp1 + temp2
  np.save("langevinEP.npy",lents)
  lFex = np.mean(vmap(vmap(lSim.Fex,in_axes = (0,None)))(xs,lts),axis = 1)
  np.save("langevinFex.npy",lFex)



