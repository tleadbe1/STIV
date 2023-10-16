import jax.numpy as np
from jax.lax import scan,cond
from jax import vmap,jit,random,grad
from functools import partial
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from bisect import bisect_left as bl



@partial(jit,static_argnums = (0,))
def rk4Step(f,x,t,dt):
  k1 = f(x,t)
  k2 = f(x + k1*dt/2,t + dt/2)
  k3 = f(x + k2*dt/2,t + dt/2)
  k4 = f(x + k3*dt,t+dt)
  return x + 1/6*dt*(k1+2*k2 + 2*k3 + k4)


def ker(x):
  return np.exp(-x@x/2)*((2*np.pi)**(-len(x)/2))

def logker(x,h):
  return -x@x/2/h**2 - len(x)/2*np.log(2*np.pi*h**2)


@jit 
def eval_kde(y,xs,h):
  logkers = vmap(logker,in_axes = (0,None))(y-xs,h)
  pref = np.max(logkers)
  logkers = logkers - pref
  return np.exp(pref)*np.mean(np.exp(logkers))

@jit
def logpdf(y,xs,h):
  logkers = vmap(logker,in_axes = (0,None))(y-xs,h)
  pref = np.max(logkers)
  logkers = logkers - pref
  return pref + np.log(np.mean(np.exp(logkers)))

@jit
def logl(ys,xs,h):
  def sf(s,y):
    s = s + logpdf(y,xs,h)
    return s,None
  S,_ = scan(sf,0.,ys)
  return S

@jit #Flag, JAX may not like this!
def dxkde(y,xs,h):
  def sf(s,z):
    s = s - z*ker(z)
    return s,None
  S,_ = scan(sf,np.zeros(len(y)),(y-xs)/h)
  return S/len(xs)/h**(len(y)+1)



@jit
def entaprx(ys,xs,h,lyCut):
  def sf(s,y):
    l = logpdf(y,xs,h)
    l = cond(np.isinf(l),lambda x: 0.,lambda x: l,None)
    s += l*(1 + np.sign(l-lyCut))/2
    return s,None
  S,_ = scan(sf,0.,ys)
  return -S/len(ys)


@partial(jit,static_argnums = (5,))
def opt(h,xmle,xkde,dh0,scale = 1/np.sqrt(2),nSteps = 100):
  def sc(c,n):
    h,v,dh = c
    vp = logl(xmle,xkde,h+dh)
    vm = logl(xmle,xkde,h-dh)
    nh,nv = cond(v < vp,lambda x: (h+dh,vp),lambda x: (h,v),None)
    nh,nv = cond(nv < vm,lambda x: (h-dh,vm),lambda x: (nh,nv),None)
    dh = cond(v == nv,lambda x: dh*scale,lambda x: dh,None)
    return (nh,nv,dh),None
  c,_ = scan(sc,(h,logl(xmle,xkde,h),dh0),None,length = nSteps)
  h,_,dh = c
  return h,dh

def indc(x):
  return 1/2*(1 + np.sign(x))

class LangevinDynamics:
  def __init__(self,k1 = 1.,l1 = 1.,k2 = 1.,l2 = 1.,eta = 1.,beta = 1.,ell = lambda t: t,nMasses = 8):
    self.k1 = k1
    self.l1 = l1
    self.k2 = k2
    self.l2 = l2
    self.h2 = k1/2*l1**2 - k2/2*l2**2
    self.eta = eta
    self.beta = beta
    self.ell = ell
    self.D = 1/self.eta/self.beta
    self.nMasses = nMasses
    return 

  def V(self,x,t):
    augx = np.zeros(x.shape[0]+2)
    augx = augx.at[1:-1].set(x)
    augx = augx.at[-1].set(self.ell(t))
    dx = augx[1:] - augx[:-1]
    Vs = (1 - indc(dx))*(self.k1/2*(dx + self.l1)**2) + indc(dx)*(self.k2/2*(dx - self.l2)**2 + self.h2)
    return np.sum(Vs)

  def F(self,x,t):
    augx = np.zeros(x.shape[0]+2)
    augx = augx.at[1:-1].set(x)
    augx = augx.at[-1].set(self.ell(t))
    dx = augx[1:] - augx[:-1]
    Fs = (1 - indc(dx))*(self.k1*(dx + self.l1)) + indc(dx)*(self.k2*(dx-self.l2))
    return Fs[1:] - Fs[:-1]
  
  def Fex(self,x,t):
    dx = self.ell(t) - x[-1]
    return (1. - indc(dx))*self.k1*(dx + self.l1) + indc(dx)*self.k2*(dx - self.l2)

  def dx(self,key,xs,t,dt):
    f = vmap(self.F,in_axes = (0,None))(xs,t)
    key,k1 = random.split(key)
    bm = random.normal(k1,xs.shape)
    return key,f*dt/self.eta + np.sqrt(2*dt/self.eta/self.beta)*bm 
    
  def dW(self,xs,t,dt):
    fex = vmap(self.Fex,in_axes = (0,None))(xs,t)
    return fex*grad(self.ell)(t)*dt


  @partial(jit,static_argnums = (0,6))
  def run(self,key,x0s,t0,W0,dt,nSteps):
    def func(carry,n):
      k0,x,t,W = carry
      k0,dx = self.dx(k0,x,t,dt)
      dW = self.dW(x,t,dt)
      carry = k0,x+dx,t+dt,W+dW
      return carry,None
    c0 = key,x0s,t0,W0
    out,_ = scan(func,c0,None,length = nSteps)
    key,x1,t1,W1 = out
    return key,x1,t1,W1

  @partial(jit,static_argnums = (0,-2,-1))
  def runCollect(self,key,x0s,t0,W0,dt,nSteps,nLeap):
    def sf(c,n):
      key,x,t,W = c
      c = self.run(key,x,t,W,dt,nLeap)
      return c,c
    c0 = key,x0s,t0,W0
    _,out = scan(sf,c0,None,length = nSteps)
    return out
  
  @partial(jit,static_argnums = (0,))
  def entropy(self,xmle,xkde,xeval,h0):
    h,_ = opt(h0,xmle,xkde,0.05,nSteps = 10)
    return entaprx(xeval,xkde,h,-np.log(len(xkde))/np.log(10)*3)
 
  @partial(jit,static_argnums = (0,)) 
  def entropyProduction(self,xmles,xkdes,xevals,ts,Ws,h0):
    xs = np.concatenate((xmles,xkdes,xevals),axis = 1)
    dt = ts[1] - ts[0]
    Vs = np.mean(vmap(vmap(self.V,in_axes = (0,None)),in_axes = (0,0))(xs,ts),axis = -1)
    Ws = np.mean(Ws,axis = -1)
    Vs = Vs.at[1:].set(Vs[1:] - Vs[:-1])/dt
    Ws = Ws.at[1:].set(Ws[1:] - Ws[:-1])/dt
    def sf(n,i):
      out = self.entropy(xmles[i],xkdes[i],xevals[i],h0)
      return None,out
    _,ents = scan(sf,None,np.arange(xmles.shape[0]))
    ents = ents.at[1:].set(ents[1:] - ents[:-1])/self.beta
    return (Ws - Vs),ents
  
##  def entropyProduction1(self,x,t):
##    n = len(x)
##    nmle = int(0.02*n)
##    nkde = int(0.44*n)
##    xmle = x[:nmle]
##    xkde = x[nmle:nmle+nkde]
##    xeval = x[nmle+nkde:]
##    h,_ = opt(0.2,xmle,xkde,0.03,nSteps = 30)
##    def sf(ep,y):
##      dadx = dxkde(y,xkde,h)/eval_kde(y,xkde,h)/self.beta - self.F(y,t)
##      ep = ep + dadx@dadx/self.eta
##      return ep,None
##    ep,_ = scan(sf,0.,xeval)
##    return ep/len(xeval)
    

  def phases(self,xs,ts):
    # ts,nSamps,dim
    augxs = np.zeros((xs.shape[0],xs.shape[1],xs.shape[2]+2))
    augxs = augxs.at[:,:,1:-1].set(xs)
    augxs = augxs.at[:,:,-1].set((np.ones((xs.shape[1],xs.shape[0]))*self.ell(ts)).T)
    dx = augxs[:,:,1:] - augxs[:,:,:-1]
    return np.sum(dx > 0.,axis = 1)/dx.shape[1]

  def getFrontLoc(self,xs,ts):
    ps = self.phases(xs,ts)
    tphase = []
    for p in ps.T:
      i = bl(p,.5)
      tphase.append(ts[i])
    return np.array(tphase)

  def getFrontLocInd(self,xs,ts,sm = 4):
    print(xs.shape)
    n = len(ts)
    Smth = np.zeros((n,n))
    for i in np.arange(-sm,sm+1):
      Smth += np.diag(np.ones(n-abs(i)),i)
    xs = xs.at[:,:,1:].set(xs[:,:,1:] - xs[:,:,:-1])
    xs = np.tensordot(Smth,xs,((1,),(0,))) 
    def findTime(x,t):
      return t[np.sum(x<0)]
    tphase = np.mean(vmap(vmap(findTime,in_axes = (-1,None)),in_axes = (-1,None))(xs,ts),axis = -1)
    return tphase
    
