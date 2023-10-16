import jax.numpy as np
import numpy as onp
from scipy.stats import linregress as lr
from jax import vmap,jit
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from bisect import bisect_left as bl


useTex = False

if useTex:
  plt.rcParams.update({"text.usetex":True})

import sys
sys.path.append("/code/DynamicsSolvers")
from eyinkSigmaInvDW import *
from langevinDW import *
import os

from basic_inputs import *
  

plot00 = True
plot01 = True
plot10 = True
plot11 = True


fExColor = "blue"
eyinkColor1 = "black"
eyinkColor2 = "tab:gray"
langColor1 = "red"
langColor2 = "tab:pink"

tab20 = cm.get_cmap("tab20")(np.linspace(0,1,20))

langDash1 = (0,(2,2))
langDash2 = (0,(1,1)) 
eyinkDash1 = "solid"
eyinkDash2 = (0,(5,1))
fExDash = "dashed" 

xlp = 2.
ylp = 4.

fs = 6
lw = .7
marksizeBg = 1
marksizeBg = 3

width = 3.425197
fig =plt.figure(figsize = (width,9*width/16),dpi = 600)


from matplotlib.gridspec import GridSpec
gs = GridSpec(4,10,figure = fig)

ax0 = fig.add_subplot(gs[:2,:5])
ax1 = fig.add_subplot(gs[:2,5:])
ax2 = fig.add_subplot(gs[2:,:5])
ax3 = fig.add_subplot(gs[2:,5:])
fig.subplots_adjust(bottom = .125,top = .88,left = .11,right = .9,wspace = 0.10,hspace = 0.06)


shearRates =  [1/16,1/8,1/4,1/2,1.,2.,4.,6.,8.,10.]
files =    ["V1_16","V1_8","V1_4","V1_2","V1","V2","V4","V6","V8","V10"]



# Comparison of 17 masses and 62 mass system

if plot00:
  os.chdir("/code/Fig5/Linear17/V1/")

  ts8 = np.load("ts.npy")
  tsE8 = np.load("tsLong.npy")
  nEnd8 = bl(ts8,tsE8[-1]) + 1
  ts8 = ts8[:nEnd8]  # Only show Langevin to same time as STIV
  ells = ell(ts8) - 2*ell(0.)
  ellsE = ell(tsE8) - 2*ell(0.)
  
  
  xs8 = np.load("xs.npy")  # The 8 is just a remnant of a previous version
  mxs8 = np.mean(xs8,axis = 1)
  mxs8 = mxs8[:nEnd8]
  l18 = l10/(mxs8.shape[-1] + 1)
  mus8 = np.load("musLong.npy")
  
  os.chdir("/code/Fig5/Linear62/V1/")

  ts64 = np.load("ts.npy")
  tsE64 = np.load("tsLong.npy")
  xs64 = np.load("xs.npy")
  mxs64 = np.mean(xs64,axis = 1)
  l164 = l10/(mxs64.shape[-1] + 1)
  mus64 = np.load("musLong.npy")

  
  
  mxs8 = mxs8 + 2*l18*np.arange(1,mxs8.shape[-1]+1)
  mxs64 = mxs64 + 2*l164*np.arange(1,mxs64.shape[-1]+1)
  mus8 = mus8 + 2*l18*np.arange(1,mus8.shape[-1]+1)
  mus64 = mus64 + 2*l164*np.arange(1,mus64.shape[-1]+1)
  
  
  mxs8 = mxs8[:,1::2]
  mus8 = mus8[:,1::2]
  
  mxs64 = mxs64[:,6::7]
  mus64 = mus64[:,6::7]
  
  
  zors = [2,3,0,1]  
  lws = [lw,lw,1.5*lw,1.5*lw]  

  for i in range(mus8.shape[-1]):
    if not i:
      ax0.plot(ts8,mxs8[:,i],color = langColor2,linestyle = langDash2,zorder = zors[1],label = "Langevin 17 masses",linewidth= lws[1])
      ax0.plot(tsE8,mus8[:,i],color = eyinkColor2,linestyle = eyinkDash2,zorder = zors[0] ,label = "STIV 17 masses",linewidth=lws[0])
    else:
      ax0.plot(ts8,mxs8[:,i],color = langColor2,linestyle = langDash2,zorder = zors[1],linewidth=lws[1])
      ax0.plot(tsE8,mus8[:,i],color = eyinkColor2,linestyle = eyinkDash2,zorder = zors[0],linewidth=lws[0])
  
  for i in range(mus64.shape[-1]):
    if not i:
      ax0.plot(ts64,mxs64[:,i],color = langColor1,linestyle = langDash1,zorder = zors[3],label = "Langevin 62 masses",linewidth=lws[3])
      ax0.plot(tsE64,mus64[:,i],color = eyinkColor1,linestyle = eyinkDash1,zorder = zors[2],label = "STIV 62 masses",linewidth=lws[2])
    else:
      ax0.plot(ts64,mxs64[:,i],color = langColor1,linestyle = langDash1,zorder = zors[3],linewidth=lws[3])
      ax0.plot(tsE64,mus64[:,i],color = eyinkColor1,linestyle = eyinkDash1,zorder = zors[2],linewidth=lws[2])
  ax0.plot(ts8,ells,color = fExColor,linestyle = fExDash,zorder = 4,label = "External protocol",linewidth=lw)
  ax0.set_xlabel("Time",fontsize = fs,labelpad = xlp)
  ax0.set_ylabel("Mean position",fontsize = fs,labelpad = 1.15*ylp)
  
  ax0.xaxis.set_label_position("top")
  ax0.tick_params(top=True,labeltop = True,bottom = False,labelbottom = False)
  


os.chdir("/code/Fig5/Linear17/")

# -------- External force vs extension for all shear rates --  ax1

if plot01:
  from matplotlib import colormaps as cm
  from bisect import bisect_left as bl
  tab20 = cm.get_cmap("tab20")(np.linspace(0,1,20))
  

  ellf = (-l10*(1 - np.load("V1/ts.npy")[-1]))
  for i,f,u in zip(np.arange(len(shearRates)),files,shearRates):
    eFex = np.load(f+ "/eyinkFexLong.npy")
    lFex = np.load(f+ "/langevinFex.npy")
    ts = np.load(f+"/ts.npy")
    tsE = np.load(f+"/tsLong.npy")
    
    ells = -l10*(1-u*ts)
    ellsE = -l10*(1-u*tsE)
    j = bl(ells,ellf) #  only show data up to the same extension
    jE = bl(ellsE,ellf)

    ax1.plot(ells[:j]+2*l10,lFex[:j],color = tab20[2*i],linestyle = "dashed",linewidth = lw,zorder = 1)
    ax1.plot(ellsE[:jE]+2*l10,eFex[:jE],color = tab20[2*i+1],linestyle = "solid",linewidth = lw,zorder = 0)
  

# --------- Phase front speed vs shear rates ------- ax2

# working from right here!


if plot10:
  os.chdir("/code/Fig5/Linear17/VelData/") 
  eVels = np.load("eynVels.npy")
  lVels = np.load("langVels.npy") 
  for eps,lps,u,i in zip(eVels,lVels,shearRates,np.arange(len(eVels))):
    ax2.plot([u,],[abs(eps),],color = tab20[2*i+1],marker = "x",markersize = marksizeBg,linewidth = 0,zorder = 0)
    ax2.plot([u,],[abs(lps),],color = tab20[2*i],marker = "+",markersize = marksizeBg,linewidth = 0,zorder = 1)

 
  
  
# --------- Entropy production of interface for all shear rates ---- #ax3

if plot11:
  from bisect import bisect_left as bl
  os.chdir("/code/Fig5/Linear17/")
  ts0 = np.load("V1/ts.npy")
  ellf = -l10*(1 - ts0[len(ts0)//2 + 1])
  axes = [fig.add_subplot(gs[2+j//5,5+j%5]) for j in range(10)]
  for f,u,i in zip(files,shearRates,np.arange(len(files))):
    eEP = np.load(f + "/eyinkEPLong.npy")
    eVEP = np.load("VisDissData/eVisDiss%s.npy" %(f))
    ets = np.load(f + "/tsLong.npy")
    lEP = np.load(f + "/langevinEP.npy")
    lVEP = np.load("VisDissData/lVisDiss%s.npy" %(f))
    lts = np.load(f + "/ts.npy")
    
    lPEP = lEP[1:] - lVEP[1:]
     
    lts = lts[:-1]
    ePEP = eEP - eVEP
    eElls = -l10*(1-u*ets)
    lElls = -l10*(1-u*lts)
    eIndx = bl(eElls,ellf)
    lIndx = bl(lElls,ellf)
  ## TAG1  
    #ax3.plot(lElls[:lIndx],lPEP[:lIndx],color = tab20[2*i],linestyle = "dashed",linewidth = lw,zorder = 1)
    #ax3.plot(eElls[:eIndx],ePEP[:eIndx],color = tab20[2*i+1],linestyle = "solid",linewidth = lw,zorder = 0)

    axes[i].plot(lElls[:lIndx]-2*lElls[0],lPEP[:lIndx],color = tab20[2*i],linestyle = "dashed",linewidth = lw,zorder = 1)
    axes[i].plot(eElls[:eIndx]-2*eElls[0],ePEP[:eIndx],color = tab20[2*i+1],linestyle = "solid",linewidth = lw,zorder = 0)

    axes[i].set_xlim([0.5,4.])
    axes[i].set_ylim([-1.,12.])

    
    if i in [4,9]:
      axes[i].yaxis.set_label_position("right")
      axes[i].tick_params(right=True,labelright=True,left=False,labelleft=False)
      axes[i].set_yticks([0.,10.])
    else:
      axes[i].set_yticks([])
      axes[i].set_yticklabels([])
    if not i in [5,6,7,8,9]:
      axes[i].set_xticks([])
      axes[i].set_xticklabels([])
    else:
      axes[i].set_xticks([1.,3.])
    





#LABELS
import matplotlib.lines as mlines

l_line = mlines.Line2D([],[],color = tab20[6],linestyle = langDash1,label = "Langevin",linewidth = lw,marker = "+",markersize = marksizeBg)
e_line = mlines.Line2D([],[],color = tab20[7],linestyle = eyinkDash1,label = "STIV",linewidth = lw,marker = "x",markersize = marksizeBg)
l_scat = mlines.Line2D([],[],color = tab20[6],label = "Langevin",linewidth = 0,marker = "+",markersize = marksizeBg)
e_scat = mlines.Line2D([],[],color = tab20[7],label = "STIV",linewidth = 0,markersize = marksizeBg,marker = "x")


ax1.legend(loc= "upper left",bbox_to_anchor = (-0.04,1.04),handles = [e_line,l_line],fontsize = 6,frameon = False,framealpha = 1.)

ax0.set_yticks([0.,2.,4.,6.,8.])
#ax0.set_yticklabels([0.,2.,4.,6.])
ax0.set_xticks([0,2.5,5.,7.5,10.])
ax0.set_xticklabels(["0","2.5","5","7.5","10"])


ax1.set_xlabel("Extension",fontsize = fs,labelpad = xlp)
if useTex:
  ax1.set_ylabel(r"$F^\textnormal{ex}$",fontsize = fs,labelpad = ylp)
else:
  ax1.set_ylabel("F_ex",fontsize = fs,labelpad = ylp)
ax1.xaxis.set_label_position("top")
ax1.tick_params(top=True,labeltop = True,bottom = False,labelbottom = False)
ax1.yaxis.set_label_position("right")
ax1.tick_params(right=True,labelright = True,left = False,labelleft = False)


ax1.set_xticks([1.,3.,5.,7.])
#ax1.set_xticklabels([1.,2.,3.,4.,5.])


ax2.set_xlabel("Strain rate",fontsize = fs,labelpad = .75*xlp)
ax2.set_ylabel("Front speed",fontsize = fs,labelpad = .75*ylp)
ax2.set_xscale("log")
ax2.set_xticks([1/16,1/8,1/4,.5,1.,2.,4.,8.])
if useTex:
  ax2.set_xticklabels([r"$\frac1{16}$",r"$\frac1{8}$",r"$\frac1{4}$",r"$\frac1{2}$",r"$1$",r"$2$",r"$4$",r"$8$"])
else:
  ax2.set_xticklabels(["1/16","1/8","1/4","1/2","1","2","4","8"])
ax2.set_xticks([],minor=True)
ax2.set_yticks([0.,.2,.4])
ax2.set_yticklabels(["0",".2",".4"])


ax3.yaxis.set_label_position("right")
ax3.set_xticks([])
ax3.set_yticks([])
#ax3.xaxis.set_ticks_position("none")
#ax3.yaxis.set_ticks_position("none")
#ax3.tick_params(right=True,labelright=True,left=False,labelleft=False)
ax3.set_xlabel("Extension",fontsize = fs,labelpad = 5*xlp)
if useTex:
  ax3.set_ylabel(r"Phase front $T\dot{S}^{\textnormal{tot}}$",fontsize = fs,labelpad = 3.3*ylp)
else:
  ax3.set_ylabel("Phase front TdS_tot/dt",fontsize = fs,labelpad = 3.3*ylp)
ax0.text(.5,.9,"A",fontsize = 1.2*fs,ha = "center",va = "center",transform = ax0.transAxes)
ax1.text(.5,.9,"B",fontsize = 1.2*fs,ha = "center",va = "center",transform = ax1.transAxes)
ax2.text(.5,.9,"C",fontsize = 1.2*fs,ha = "center",va = "center",transform = ax2.transAxes)
#ax3.text(.5,.9,"D",fontsize = 1.2*fs,ha = "center",va = "center",transform = #ax3.transAxes)
axes[2].text(.5,.825,"D",fontsize = 1.2*fs,ha = "center",va = "center",transform = axes[2].transAxes)


os.chdir("/code/Fig5")

fig.savefig("Fig5.pdf")


