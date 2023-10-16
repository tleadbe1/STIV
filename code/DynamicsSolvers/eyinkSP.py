import jax.numpy as np
from jax import random,jit,vmap,grad,jacrev
from jax.scipy.stats.norm import pdf,cdf
from jax.scipy.linalg import solve_triangular as solve_tri 
from jax.lax import scan
from functools import partial

solveU= jit(lambda U,b: solve_tri(U,b,lower=False))
solveL=jit(lambda L,b:solve_tri(L,b,lower = True))
choSolve = jit(lambda L,b: solve_tri(L.T,solve_tri(L,b,lower = True),lower = False))


@jit
def bDiff(x):
  x = x.at[1:].set(x[1:]-x[:-1])
  return x

@jit
def fDiff(x):
  x = x.at[:-1].set(x[1:]-x[:-1])
  x = x.at[-1].set(-x[-1])
  return x

@jit
def diff2(x):
  x0 = x[0]
  x = bDiff(fDiff(x))
  x = x.at[0].set(x[0] - x0)
  return x


@partial(jit,static_argnums = (0,))
def rk4Step(f,x,y,t,dt):
  kx1,ky1 = f(x,y,t)
  kx2,ky2 = f(x + kx1*dt/2,y + ky1*dt/2,t + dt/2)
  kx3,ky3 = f(x + kx2*dt/2,y + ky2*dt/2,t + dt/2)
  kx4,ky4 = f(x + kx3*dt,y + ky3*dt,t + dt)
  return x + dt/6 *(kx1 + 2*kx2 + 2*kx3 + kx4),y + dt/6 *(ky1 + 2*ky2 + 2*ky3 + ky4)

class EyinkSim:
  def __init__(self, k = 1., l= 1.,eta = 1., beta = 1. , ell = lambda t: t,nMasses = 2):
    self.l = l
    self.k = k
    self.eta = eta
    self.beta = beta
    self.D = 1/eta/beta
    self.ell = ell
    self.nMasses = nMasses
    return
 

  def alphaDot(self,mu,s,t):
    muDot = -self.k/self.eta*(mu - self.ell(t))
    sDot = -self.k/self.eta*s*(1 - 1/self.k/self.beta/s**2)
    return muDot,sDot

  @partial(jit,static_argnums  =(0,))
  def step(self,mu,s,t,dt):
    mu,s = rk4Step(self.alphaDot,mu,s,t,dt)
    return mu,s,t+dt

  def run(self,mu,s,t,dt,nSteps=1):
    def sf(c,n):
      mu,s,t = c
      c = self.step(mu,s,t,dt)
      return c,None
    out,_ = scan(sf,(mu,s,t),None,length = nSteps)
    return out
  
  def Fex(self,mu,s,t): 
    return self.k*(self.ell(t) - mu)

  @partial(jit,static_argnums = (0,))
  def entropyProduction(self,mu,s,t):
    muDot = -self.k/self.eta*(mu - self.ell(t))
    sDot = -self.k/self.eta*s*(1 - 1/self.k/self.beta/s**2)
    return self.eta*(muDot**2 + sDot**2)

