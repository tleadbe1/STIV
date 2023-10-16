import jax.numpy as np
from jax.lax import scan,cond
from jax import vmap,jit,random,grad
from functools import partial
import matplotlib.pyplot as plt
from matplotlib import cm
from bisect import bisect_left as bl




class LangevinDynamics:
  def __init__(self,k = 1.,l = 1.,eta = 1.,beta = 1.,ell = lambda t: t,nMasses = 8):
    self.k = k
    self.l = l
    self.eta = eta
    self.beta = beta
    self.ell = ell
    self.D = 1/self.eta/self.beta
    self.nMasses = nMasses
    return 

  def V(self,x,t):
    return self.k/2*(x - self.ell(t))**2

  def F(self,x,t):
    return self.k*(self.ell(t) - x)
  
  def Fex(self,x,t):
    return self.k*(self.ell(t) - x)

  def dx(self,key,xs,t,dt):
    f = vmap(self.F,in_axes = (0,None))(xs,t)
    key,k1 = random.split(key)
    bm = random.normal(k1,xs.shape)
    return key,f*dt/self.eta + np.sqrt(2*self.D*dt)*bm 
    
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
  
  
  def entropyProduction(self,xs,ts):
    mus = np.mean(xs,axis = 1)
    sigmas = np.std(xs,axis = 1)
    muDot = -self.k/self.eta *(mus - self.ell(ts))
    sigmaDot = -self.k/self.eta*sigmas*(1 - 1/self.k/self.beta/sigmas**2)
    return self.eta*(muDot**2 + sigmaDot**2) 
    
