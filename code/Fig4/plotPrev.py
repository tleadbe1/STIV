import jax.numpy as np
from jax import vmap
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import matplotlib.gridspec as gridspec
import sys
sys.path.append("/code/DynamicsSolvers")
from eyinkSigmaInvDW import *
from langevinDW import *
from inputs import *

import os
os.chdir("/data/Fig4/")

# Something is up with the entropy production but otherwise this is looking great!


useTex = False


if useTex:
  plt.rcParams["text.usetex"] = True





eSim = EyinkSim(k1 = k1,l1 = l1,k2 = k2,l2 = l2,beta = beta,eta = eta,ell = ell,nMasses = nMasses)
lSim = LangevinDynamics(l1=l1,k1=k1,k2 = k2,l2=l2,eta = eta,beta = beta,ell = ell,nMasses = nMasses)


eyinkColor = "black"
eyinkDash = "solid"

langColor = "red"
langDash = (0,(2,2))

fExColor = "blue"
fExDash = "dashed"

lw = 1.
fs = 8
labelfs = 8

import os
os.chdir("Linear")

mus = np.load("mus.npy")
Ss = np.load("SInvs.npy")
ts = np.load("ts.npy")
xs = np.load("xs.npy")
eents00 = np.load("stivEP.npy")
lents00 = np.load("langevinEP.npy")
lFex00 = np.load("langevinFex.npy")
eFex00 = np.load("stivFex.npy")

os.chdir("../SineSlow")

eents01 = np.load("stivEP.npy")
lents01 = np.load("langevinEP.npy")
lFex01 = np.load("langevinFex.npy")
eFex01 = np.load("stivFex.npy")

os.chdir("../Step")

eents10 = np.load("stivEP.npy")
lents10 = np.load("langevinEP.npy")
lFex10 = np.load("langevinFex.npy")
eFex10 = np.load("stivFex.npy")


os.chdir("../OffSet")

eents11 = np.load("stivEP.npy")
lents11 = np.load("langevinEP.npy")
lFex11 = np.load("langevinFex.npy")
eFex11 = np.load("stivFex.npy")

os.chdir("../")

Sigmas = vmap(np.linalg.inv)(Ss)
stds = np.sqrt(vmap(np.diag)(Sigmas))

mxs = np.mean(xs,axis = 1)
stdxs = np.std(xs,axis = 1)

PNASFigSize = (7.007874,3.941929125)
PosterFigSize = (10,10*9/16)
fig = plt.figure(figsize = PNASFigSize,dpi = 700)
gs = gridspec.GridSpec(4,4)
ax0 = fig.add_subplot(gs[:2,:2])
ax1 = fig.add_subplot(gs[2:,:2])
ax = [ax0,ax1]

fex00 = fig.add_subplot(gs[0,2])
fex01 = fig.add_subplot(gs[0,3])
fex10 = fig.add_subplot(gs[1,2])
fex11 = fig.add_subplot(gs[1,3])
fex = [[fex00,fex01],[fex10,fex11]]

ep00=fig.add_subplot(gs[2,2])
ep01=fig.add_subplot(gs[2,3])
ep10=fig.add_subplot(gs[3,2])
ep11=fig.add_subplot(gs[3,3])
ep = [[ep00,ep01],[ep10,ep11]]

allax = ax + fex[0] + fex[1] + ep[0] + ep[1]





eSim = EyinkSim(k1 = k1,l1 = l1,k2 = k2,l2 = l2,beta = beta,eta = eta,ell = ell,nMasses = nMasses)
lSim = LangevinDynamics(k1=k1,l1=l1,k2=k2,l2=l2,eta = eta,beta = beta,ell = ell,nMasses = nMasses)

n = int(len(ts))#int(len(ts)*7/16)

os.chdir("Linear")

mus_phase = np.load("mus.npy")[:n]
Ss_phase = np.load("SInvs.npy")[:n]
Ws_phase = np.load("Ws.npy")[:n]
xs_phase = np.load("xs.npy")[:n]
ts_phase = np.load("ts.npy")[:n]

os.chdir("../")



ePhases = eSim.phases(mus_phase,Ss_phase,ts_phase)
lPhases = lSim.phases(xs_phase,ts_phase)







distNorm = (nMasses + 1)*l1
stdNorm = 1/np.sqrt(k1*beta)

for i in range(mus.shape[-1]):
    if i == 0:
      ax[0].plot(ts,mxs[:,i]/distNorm,color = langColor,linestyle = langDash,label = "Langevin",zorder=1,linewidth = lw)
      ax[0].plot(ts,mus[:,i]/distNorm,color = eyinkColor,linestyle = eyinkDash,label = "STIV",zorder = 0,linewidth = lw)
      ax[1].plot(ts_phase,lPhases[:,i],color = langColor,linestyle = langDash,label = "Langevin",linewidth = lw,zorder = 1)
      ax[1].plot(ts_phase,ePhases[:,i],color = eyinkColor,linestyle = eyinkDash, label = "STIV",linewidth = lw,zorder = 0)
    else:
      ax[0].plot(ts,mxs[:,i]/distNorm,color = langColor,linestyle = langDash,zorder = 1,linewidth = lw)
      ax[0].plot(ts,mus[:,i]/distNorm,color = eyinkColor,linestyle = eyinkDash,zorder = 0,linewidth = lw)
      ax[1].plot(ts_phase,lPhases[:,i],color = langColor,linestyle = langDash,linewidth = lw,zorder = 1)
      ax[1].plot(ts_phase,ePhases[:,i],color = eyinkColor,linestyle = eyinkDash,linewidth = lw,zorder = 0)

ax[0].plot(ts,lin_ell(ts)/distNorm,color = fExColor,linestyle = fExDash,label = "External\nprotocol",linewidth = lw)
ax[1].plot([],[],color = fExColor,linestyle = fExDash,label = "External\nprotocol",linewidth = lw)


ax[0].legend(loc = (.12,8/16),fontsize = 8,frameon = False)

if useTex:
  ax[0].set_ylabel(r"$\mu \approx \langle x \rangle$",fontsize = labelfs)
  ax[1].set_ylabel(r"$\hat{\Phi}_i \approx \Big\langle (D_1x)_i > 0\Big\rangle$",fontsize = labelfs)
else:
  ax[0].set_ylabel("mu ~ <x>",fontsize = labelfs)
  ax[1].set_ylabel(" Phi ~ <(Dx > 0)>",fontsize = .8*labelfs)



  


fex00.plot(ts,lFex00,color = langColor,linestyle = langDash,zorder = 1,linewidth = lw)
fex00.plot(ts,eFex00,color = eyinkColor,linestyle = eyinkDash,zorder = 0,linewidth = lw)
fex01.plot(ts,lFex01,color = langColor,linestyle = langDash,zorder = 1,linewidth = lw)
fex01.plot(ts,eFex01,color = eyinkColor,linestyle = eyinkDash,zorder = 0,linewidth = lw)
fex10.plot(ts,lFex10,color = langColor,linestyle = langDash,zorder =1 ,linewidth = lw)
fex10.plot(ts,eFex10,color = eyinkColor,linestyle = eyinkDash,zorder = 0,linewidth = lw)
fex11.plot(ts,lFex11,color = langColor,linestyle = langDash,zorder = 1,linewidth = lw)
fex11.plot(ts,eFex11,color = eyinkColor,linestyle = eyinkDash, zorder = 0,linewidth = lw)



# Since the entropy prodcution is computed via a change in entropy for the Langevin dynamics, the 
# first term in the list is not use (eg lents[1:] ~ entropy[1:] - entropy[:-1])

ep00.plot(ts,eents00,color = eyinkColor,linestyle = eyinkDash,zorder = 0,linewidth = lw)
ep00.plot(ts[:-1],lents00[1:],color = langColor,linestyle = langDash,zorder = 1,linewidth = lw)
ep01.plot(ts,eents01,color = eyinkColor,linestyle = eyinkDash,zorder = 0,linewidth = lw)
ep01.plot(ts[:-1],lents01[1:],color = langColor,linestyle = langDash,zorder = 1,linewidth = lw)
ep10.plot(ts,eents10,color = eyinkColor,linestyle = eyinkDash,zorder = 0,linewidth = lw)
ep10.plot(ts[:-1],lents10[1:],color = langColor,linestyle = langDash,zorder = 1,linewidth = lw)
ep11.plot(ts,eents11,color = eyinkColor,linestyle = eyinkDash,zorder = 0,linewidth = lw)
ep11.plot(ts[:-1],lents11[1:],color = langColor,linestyle = langDash,zorder = 1,linewidth = lw)


ax[1].set_xlabel("Time",fontsize = labelfs)
ep10.set_xlabel("Time",fontsize = labelfs)
ep11.set_xlabel("Time",fontsize = labelfs)
ep11.yaxis.set_label_position("right")
ep01.yaxis.set_label_position("right")
fex01.yaxis.set_label_position("right")
fex11.yaxis.set_label_position("right")
if useTex:
  ep11.set_ylabel(r"$T\frac{\textnormal{d}\hat{S}}{\textnormal{d}t}^{\textnormal{tot}}$",fontsize = labelfs)
  ep01.set_ylabel(r"$T\frac{\textnormal{d}\hat{S}}{\textnormal{d}t}^{\textnormal{tot}}$",fontsize = labelfs)
  fex01.set_ylabel(r"$\hat{F}^{\textnormal{ex}}$",fontsize = labelfs)
  fex11.set_ylabel(r"$\hat{F}^{\textnormal{ex}}$",fontsize = labelfs)
else:
  ep11.set_ylabel("T dS_tot/dt",fontsize = labelfs)
  ep01.set_ylabel("T dS_tot/dt",fontsize = labelfs)
  fex01.set_ylabel("F_ex",fontsize = labelfs)
  fex11.set_ylabel("F_ex",fontsize = labelfs)
##ep00.set_title("linear",fontsize = 8)
##ep01.set_title("Constant",fontsize = 8)
##ep10.set_title("Sine",fontsize = 8)
##ep11.set_title("Step",fontsize = 8)
##
##fex00.set_title("linear",fontsize = 8)
##fex01.set_title("Constant",fontsize = 8)
##fex10.set_title("Sine",fontsize = 8)
##fex11.set_title("Step",fontsize = 8)



# system image inset
import matplotlib.image as mpimg
sysImg = mpimg.imread("/code/Fig4/MassSpringChainBrighter.tif")


axImInset = ax[0].inset_axes([.55,10/16+0.06,.4,.27])
axImInset.imshow(sysImg)
axImInset.set_xticks([])
axImInset.set_xticklabels([])
axImInset.set_yticks([])
axImInset.set_yticklabels([])

axImInset.text(.9,.75,"E",fontsize = 12,ha = "center",va = "center",transform = axImInset.transAxes)





fexBox = [.01,.76,.25,.25]

fex00inset = fex00.inset_axes(fexBox)
fex00inset.plot(ts,lin_ell(ts),color = fExColor,linestyle = fExDash,linewidth = lw)
fex00inset.set_xticks([])
fex00inset.set_xticklabels([])
fex00inset.set_yticks([])
fex00inset.set_yticklabels([])


fex01inset = fex01.inset_axes(fexBox)
fex01inset.plot(ts,sin_ell(ts),color = fExColor,linestyle = fExDash,linewidth = lw)
fex01inset.set_xticks([])
fex01inset.set_xticklabels([])
fex01inset.set_yticks([])
fex01inset.set_yticklabels([])

fex10inset = fex10.inset_axes(fexBox)
fex10inset.plot(ts,step_ell(ts),color = fExColor,linestyle = fExDash,linewidth = lw)
fex10inset.set_xticks([])
fex10inset.set_xticklabels([])
fex10inset.set_yticks([])
fex10inset.set_yticklabels([])

fex11inset = fex11.inset_axes(fexBox)
fex11inset.plot(ts,step_ell(ts),color = fExColor,linestyle = fExDash,linewidth = lw)
fex11inset.set_xticks([])
fex11inset.set_xticklabels([])
fex11inset.set_yticks([])
fex11inset.set_yticklabels([])

epBox = [0.01,.76,.25,.25]

ep00inset = ep00.inset_axes(epBox)
ep00inset.plot(ts,lin_ell(ts),color = fExColor,linestyle = fExDash,linewidth = lw)
ep00inset.set_xticks([])
ep00inset.set_xticklabels([])
ep00inset.set_yticks([])
ep00inset.set_yticklabels([])

ep01inset = ep01.inset_axes(epBox)
ep01inset.plot(ts,sin_ell(ts),color = fExColor,linestyle = fExDash,linewidth = lw)
ep01inset.set_xticks([])
ep01inset.set_xticklabels([])
ep01inset.set_yticks([])
ep01inset.set_yticklabels([])

ep10inset = ep10.inset_axes(epBox)
ep10inset.plot(ts,step_ell(ts),color = fExColor,linestyle = fExDash,linewidth = lw)
ep10inset.set_xticks([])
ep10inset.set_xticklabels([])
ep10inset.set_yticks([])
ep10inset.set_yticklabels([])

ep11inset = ep11.inset_axes(epBox)
ep11inset.plot(ts,step_ell(ts),color = fExColor,linestyle = fExDash,linewidth = lw)
ep11inset.set_xticks([])
ep11inset.set_xticklabels([])
ep11inset.set_yticks([])
ep11inset.set_yticklabels([])

insets = [fex00inset,fex01inset,fex10inset,fex11inset,ep00inset,ep01inset,ep10inset,ep11inset]

for in_ax in insets:
  ylim = in_ax.get_ylim()
  dy = 0.1*(ylim[1] - ylim[0])
  in_ax.set_ylim([ylim[0] -dy,ylim[1]+dy])



ax[0].text(0.025,.95,"A",fontsize = 12,ha = "left",va = "top",transform = ax[0].transAxes)
ax[1].text(.025,.95,"B",fontsize = 12,ha = "left",va = "top",transform = ax[1].transAxes)
ep00.text(.55,0.95,"D1",fontsize = 12,ha = "center",va = "top",transform = ep00.transAxes)
ep01.text(.55,0.95,"D2",fontsize = 12,ha = "center",va = "top",transform = ep01.transAxes)
ep10.text(.55,0.95,"D3",fontsize = 12,ha = "center",va = "top",transform = ep10.transAxes)
ep11.text(.55,0.95,"D4",fontsize = 12,ha = "center",va = "top",transform=ep11.transAxes)
fex00.text(.55,.025,"C1",fontsize = 12,ha = "center",va = "bottom",transform = fex00.transAxes)
fex01.text(.55,.025,"C2",fontsize = 12,ha = "center",va = "bottom",transform = fex01.transAxes)
fex10.text(.55,.025,"C3",fontsize = 12,ha = "center",va = "bottom",transform = fex10.transAxes)
fex11.text(.55,.025,"C4",fontsize = 12,ha = "center",va = "bottom",transform = fex11.transAxes)


os.chdir("/code/Fig4")

fig.savefig("Fig4.pdf")

