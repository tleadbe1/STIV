import jax.numpy as np
from jax import vmap
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
import numpy as onp

import sys
sys.path.append("/code/DynamicsSolvers")
from eyinkSigmaInvDW import *
from langevinDW import *

from inputs import *

useTex = False

import os
os.chdir("/data/Fig3/")

if useTex:
  plt.rcParams["text.usetex"] = True


fs = 8
labelfs = 8
lp = 1.2
plotcmap = "YlOrRd"
lw = .29
bot = .175



def mvNormal(y,mu,L):
  return np.exp(-1/2*(y-mu)@L@L.T@(y - mu))/(2*np.pi)**(len(mu)/2) * np.abs(np.prod(np.diag(L)))


eSim = EyinkSim(k1 = k1,l1 = l1,k2 = k2,l2=l2,beta = beta,eta = eta,ell = ell,nMasses = nMasses)
lSim = LangevinDynamics(k1=k1,l1=l1,k2=k2,l2=l2,eta = eta,beta = beta,ell = ell,nMasses = nMasses)



xs = np.load("xs.npy")
mus = np.load("mus.npy")
Ss = np.load("SInvs.npy")
Ls = vmap(np.linalg.cholesky)(Ss)
ts = np.load("ts.npy")
D1 = np.array([[1.,0.],[-1.,1.]])
temp = lambda L: np.linalg.cholesky(np.linalg.inv(D1 @ np.linalg.inv(L@L.T) @ D1.T))
dLs = vmap(temp)(Ls)
dmus = mus
dmus = dmus.at[:,1:].set(dmus[:,1:]-dmus[:,:-1])
dxs = xs.at[:,:,1:].set(xs[:,:,1:] - xs[:,:,:-1])

indxs = np.arange(5,158,11)
nRow = 1
nCol = len(indxs)//nRow

bins = [np.linspace(-4,3,250),np.linspace(-4,3.5,250)]

wPNAS = 3.425197
wPNASwide = 7.007874
wPoster = 10
w = wPNASwide
h = 5/16*wPNASwide

figtemp,axtemp = plt.subplots(nRow,nCol,figsize = (w,h),dpi = 60,squeeze = False)
figtemp.subplots_adjust(left = 0.02,right = 0.985,top = .955,bottom = bot,wspace = 0.1,hspace = 0.06)

levels = []

for j,i in enumerate(indxs):
  axtemp[j//nCol,j%nCol].hist2d(onp.array(dxs[i,:,0]),onp.array(dxs[i,:,1]),bins = bins,density = True,cmap = "terrain")
  if j//nCol == 0:
    axtemp[0,j%nCol].set_xticks([])
    axtemp[0,j%nCol].set_xticklabels([])
  if not (j%nCol == 0):
    axtemp[j//nCol,j%nCol].set_yticks([])
    axtemp[j//nCol,j%nCol].set_yticklabels([])

XS,YS = np.meshgrid(np.linspace(bins[0][0],bins[0][-1],400),np.linspace(bins[1][0],bins[1][-1],400))
pnts = np.stack((XS.flatten(),YS.flatten()),axis = 1)
for j,i in enumerate(indxs):
  ZS = (vmap(mvNormal,in_axes = (0,None,None))(pnts,dmus[i],dLs[i])).reshape(XS.shape)
  out = axtemp[j//nCol,j%nCol].contour(XS,YS,ZS,colors = "k",levels = 6,linewidths = lw)
  levels.append(out.levels[2:])
  if not (j//nCol == (nRow - 1)):
    axtemp[j//nCol,j%nCol].set_xticks([])
    axtemp[j//nCol,j%nCol].set_xticklabels([])
  if not(j%nCol == 0):
    axtemp[j//nCol,j%nCol].set_yticks([])
    axtemp[j//nCol,j%nCol].set_yticklabels([])    

for j in range(nCol):
  axtemp[nRow-1,j].set_xticks([-2.,2.])
  axtemp[nRow-1,j].set_xticklabels(["-2","2"],fontsize = fs)
for i in range(nRow):
  axtemp[i,0].set_yticks([-3,0,3])
  axtemp[i,0].set_yticklabels(["-3","0","3"],fontsize = fs)


plt.close(figtemp)



from matplotlib import colormaps as cm





fig,ax = plt.subplots(nRow,nCol,figsize = (w,h),dpi = 1200,squeeze = False)
fig.subplots_adjust(left = 0.06,right = 0.985,top = .955,bottom = bot,wspace = 0.1,hspace = 0.06)


for j,i in enumerate(indxs):
  ax[j//nCol,j%nCol].hist2d(onp.array(dxs[i,:,0]),onp.array(dxs[i,:,1]),bins = bins,density = True,cmap = "terrain",rasterized = True)
  #if j//nCol == 0:
  #  ax[0,j%nCol].set_xticks([])
  #  ax[0,j%nCol].set_xticklabels([])
  #if not (j%nCol == 0):
  #  ax[j//nCol,j%nCol].set_yticks([])
  #  ax[j//nCol,j%nCol].set_yticklabels([])

XS,YS = np.meshgrid(np.linspace(bins[0][0],bins[0][-1],400),np.linspace(bins[1][0],bins[1][-1],400))
pnts = np.stack((XS.flatten(),YS.flatten()),axis = 1)
for j,i in enumerate(indxs):
  ZS = (vmap(mvNormal,in_axes = (0,None,None))(pnts,dmus[i],dLs[i])).reshape(XS.shape)
  out = ax[j//nCol,j%nCol].contour(XS,YS,ZS,cmap = plotcmap,levels = levels[j],linewidths = lw)
  #if not (j//nCol == (nRow - 1)):
  #  ax[j//nCol,j%nCol].set_xticks([])
  #  ax[j//nCol,j%nCol].set_xticklabels([])
  #if not(j%nCol == 0):
  #  ax[j//nCol,j%nCol].set_yticks([])
  #  ax[j//nCol,j%nCol].set_yticklabels([])    

for j in range(nCol):
  ax[0,j].set_xticks([-2.,2.])
  ax[0,j].set_xticklabels(["-2","2"],fontsize = fs)
  if not j == 0:
    ax[0,j].set_yticks([])
  if j%2:
    if j==7:
      if useTex:
        ax[0,j].text(-4,-5.5,r"$\longrightarrow\quad t\quad \longrightarrow $",fontsize = 12,ha = "center")
      else:
        ax[0,j].text(-4,-5.5,"-->   t   --->",fontsize = 12,ha = "center")
  else:
    if useTex:
      ax[0,j].set_xlabel(r"$x_1$",fontsize = labelfs,labelpad = lp)
    else:
      ax[0,j].set_xlabel("x_1",fontsize = labelfs,labelpad = lp)

for i in range(nRow):
  ax[i,0].set_yticks([-3,0,3])
  ax[i,0].set_yticklabels(["-3","0","3"],fontsize = fs)





if useTex:
  ax[0,0].set_ylabel(r"$x_2 - x_1$",fontsize = labelfs,labelpad = lp)
else:
  ax[0,0].set_ylabel("x_2 - x_1",fontsize = labelfs,labelpad = lp)
#ax[,0].set_ylabel("Length of spring 2",fontsize = fs,labelpad = lp) 
#ax[0,2].set_xlabel(r"$x_1$",fontsize = fs,labelpad = lp)
#ax[0,8].set_xlabel(r"$x_1$",fontsize = fs,labelpad = lp)
#ax[0,11].set_xlabel(r"$x_1$",fontsize = fs,labelpad = lp)
#ax[0,3].xaxis.set_label_coords(-.025,-.225)
#tax0 = ax[0,-1].twinx()
#tax0.set_ylabel("Langevin density",fontsize = fs,labelpad = lp)
#tax1 = ax[1,-1].twinx()
#tax1.set_ylabel("STIV density",fontsize = fs,labelpad = lp)
#tax0.set_yticks([])
#tax1.set_yticks([])

os.chdir("/code/Fig3/")

fig.savefig("Fig3.pdf")

