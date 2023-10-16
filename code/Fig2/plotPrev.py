import jax.numpy as np
from jax import vmap
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec

import sys
sys.path.append("/code/DynamicsSolvers")
from eyinkSP import *
from langevinSP import *


from inputs import *

import os

os.chdir("/data/Fig2/")


useTex = False

if useTex:
    plt.rcParams.update({"text.usetex":True})

eSim = EyinkSim(k = k,beta = beta,l = l,eta = eta,ell = ell,nMasses = nMasses)
lSim = LangevinDynamics(l=l,k=k,eta = eta,beta = beta,ell = ell,nMasses = nMasses)

xlp = 4.
ylp = 3.

eyinkColor = "black"
eyinkDash = "solid"

langColor = "red"
langDash = (0,(2,2))

fExColor = "blue"
fExDash = "dashed"

lw = 1.
fs = 8


mus = np.load("mus.npy")
sigs = np.load("sigs.npy")
ts = np.load("ts.npy")
xs = np.load("xs.npy")
eents = np.load("stivEP.npy")
lents = np.load("langevinEP.npy")
ells = lSim.ell(ts)

lFex = np.load("langevinFex.npy")
eFex = np.load("stivFex.npy")

mxs = np.mean(xs,axis = -1)
vxs = np.std(xs,axis = -1)
pnasSize = (3.425197,3.425197*9/16)
dodSize = (10,90/16)
size = pnasSize
fig,axes  = plt.subplots(2,2,figsize = size ,dpi = 900)
fig.subplots_adjust(wspace = 0.06,hspace = 0.06)

axes[0,0].plot(ts,mxs,color = langColor,linestyle = langDash,zorder = 1,linewidth = lw)
axes[0,0].plot(ts,mus,color = eyinkColor,linestyle = eyinkDash,zorder = 0,linewidth = lw)
axes[0,0].plot(ts,ells,color = fExColor,linestyle = fExDash,zorder = 1,linewidth = lw)
axes[1,0].plot(ts,vxs,color = langColor,linestyle = langDash,zorder = 1,linewidth = lw,label = "Langevin")
axes[1,0].plot(ts,sigs,color = eyinkColor,linestyle = eyinkDash,zorder = 0,linewidth = lw,label = "STIV")
axes[1,0].plot([],[],color = fExColor,linestyle = fExDash,zorder = 1,linewidth = lw,label = "External\nprotocol")
axes[1,0].legend(fontsize = fs*.6,loc = "center right",frameon = False)
axes[0,1].plot(ts,lFex,color = langColor,linestyle = langDash,zorder =1,linewidth = lw)
axes[0,1].plot(ts,eFex,color = eyinkColor,linestyle = eyinkDash,zorder =0,linewidth = lw)
axes[1,1].plot(ts,lents,color = langColor,linestyle = langDash,zorder =1,linewidth = lw)
axes[1,1].plot(ts,eents,color = eyinkColor,linestyle = eyinkDash,zorder =0,linewidth = lw)

axes[1,0].set_xlabel("Time",fontsize = fs)
axes[1,1].set_xlabel("Time",fontsize = fs)


if useTex:
    axes[0,0].set_ylabel(r"$\mu \approx \langle x \rangle$",fontsize = fs,labelpad = ylp)
    axes[1,0].set_ylabel(r"$\sigma \approx \sqrt{\langle (x - \langle x\rangle )^2 \rangle}$",fontsize = fs,labelpad = ylp)
    axes[0,1].set_ylabel(r"$F^\textnormal{ex} \approx \frac{\partial\hat{A}}{\partial \lambda}^{\textnormal{neq}}$",fontsize = fs,labelpad = ylp)
    axes[1,1].set_ylabel(r"$T\frac{\textnormal{d}\hat{S}}{\textnormal{d}t}^{\textnormal{tot}}$", fontsize = fs,labelpad = ylp)
else:
    axes[0,0].set_ylabel("mu",fontsize = fs,labelpad = ylp)
    axes[1,0].set_ylabel("sigma",fontsize = fs,labelpad = ylp)
    axes[0,1].set_ylabel("F_ex",fontsize = fs,labelpad = ylp)
    axes[1,1].set_ylabel("T * dS_tot/dt", fontsize = .7*fs,labelpad = ylp)





axes[0,0].xaxis.set_label_position("top")
axes[0,1].xaxis.set_label_position("top")
axes[0,1].yaxis.set_label_position("right")
axes[1,1].yaxis.set_label_position("right")
axes[0,0].tick_params(top=True,labeltop = True,bottom = False,labelbottom = False)
axes[0,1].tick_params(top=True,labeltop = True,bottom = False,labelbottom = False)
axes[0,1].tick_params(right=True,labelright = True,left = False,labelleft = False)
axes[1,1].tick_params(right=True,labelright = True,left = False,labelleft = False)

axes[0,0].set_xticks([])
axes[0,1].set_xticks([])

axes[0,0].text(.5,.9,"A",fontsize = 1.2*fs,ha = "center",va = "center",transform = axes[0,0].transAxes)
axes[0,1].text(.5,.9,"C",fontsize = 1.2*fs,ha = "center",va = "center",transform = axes[0,1].transAxes)
axes[1,0].text(.5,.9,"B",fontsize = 1.2*fs,ha = "center",va = "center",transform = axes[1,0].transAxes)
axes[1,1].text(.5,.9,"D",fontsize = 1.2*fs,ha = "center",va = "center",transform = axes[1,1].transAxes)

fig.tight_layout()
os.chdir("/code/Fig2/")

fig.savefig("Fig2.pdf")

