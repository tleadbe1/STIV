import jax.numpy as np
from jax import random,jit,vmap,grad,jacrev
from jax.scipy.stats.norm import pdf,cdf
from jax.scipy.linalg import solve_triangular as solve_tri 
from jax.lax import scan
from functools import partial
from bisect import bisect_left as bl

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
  def __init__(self, k1 = 1., l1= 1.,k2 = 1.,l2 = 1.,eta = 1., beta = 1. , ell = lambda t: t,nMasses = 2):
    self.l1 = l1
    self.k1 = k1
    self.k2 = k2
    self.l2 = l2
    self.h2 = k1/2.*l1**2 - k2/2.*l2**2
    self.eta = eta
    self.beta = beta
    self.D = 1/eta/beta
    self.ell = ell
    self.nMasses = nMasses
    return
 
  def sample(self,key,mu,Sinv,n=1):
    L = np.linalg.cholesky(Sinv)
    k0,k1 = random.split(key)
    us = random.normal(k1,(n,mu.shape[0]))
    x = solveU(L.T,us.T).T + mu
    return k0,x

  def getSigma(self,Sinv):
    L = np.linalg.cholesky(Sinv)
    return choSolve(L,np.eye(len(L)))
    

  def alphaDot(self,mu,Sinv,t):
    L = np.linalg.cholesky(Sinv)
    dmu = bDiff(mu)
    dmuLambda = self.ell(t) - mu[-1]
    D1 = np.eye(self.nMasses) - np.diag(np.ones(self.nMasses -1),-1)
    stds = np.sqrt(np.diag(D1 @ choSolve(L,D1.T)))
    stdLambda = abs(1/L[-1,-1])
  
    mMu = (self.k1*(dmu + self.l1)*cdf(-dmu/stds) +
            self.k2*(dmu - self.l2)*cdf(dmu/stds) +
            (self.k2 - self.k1)*stds*pdf(dmu/stds) )
    mLambda = (self.k1*(dmuLambda + self.l1)*cdf(-dmuLambda/stdLambda) +
            self.k2*(dmuLambda - self.l2)*cdf(dmuLambda/stdLambda) +
            (self.k2 - self.k1)*stdLambda*pdf(dmuLambda/stdLambda) )

    muDot = fDiff(mMu)/self.eta
    muDot = muDot.at[-1].set(muDot[-1] + mLambda/self.eta)
    wL = (self.k1*cdf(-dmu/stds) + self.k2*cdf(dmu/stds)
           -(self.k1*self.l1 + self.k2*self.l2)*pdf(dmu/stds)/stds )
    wLambda = (self.k1*cdf(-dmuLambda/stdLambda)
                +self.k2*cdf(dmuLambda/stdLambda)
                -(self.k1*self.l1 + self.k2*self.l2)*pdf(dmuLambda/stdLambda)/stdLambda )
    D2w = -(D1.T*wL) @ D1
    D2w = D2w.at[-1,-1].set(D2w[-1,-1] - wLambda)
    SinvDot = -2/self.eta*((D2w@Sinv + Sinv@D2w)/2 + Sinv@Sinv/self.beta)
    return muDot,SinvDot

  @partial(jit,static_argnums  =(0,))
  def step(self,mu,Sinv,t,dt):
    mu,Sinv = rk4Step(self.alphaDot,mu,Sinv,t,dt)
    return mu,Sinv,t+dt

  @partial(jit,static_argnums = (0,-1))
  def run(self,mu,Sinv,t,dt,nSteps=1):
    def sf(c,n):
      mu,Sinv,t = c
      c = self.step(mu,Sinv,t,dt)
      return c,None
    out,_ = scan(sf,(mu,Sinv,t),None,length = nSteps)
    return out

  @partial(jit,static_argnums = (0,-2,-1))
  def runCollect(self,mu,Sinv,t,dt,nSteps,nLeap):
    def sf(c,n):
      mu,Sinv,t = c
      c = self.run(mu,Sinv,t,dt,nLeap)
      return c,c
    _,out = scan(sf,(mu,Sinv,t),None,length = nSteps)
    return out
  
  def Fex(self,mu,Sinv,t):
    L = np.linalg.cholesky(Sinv) 
    m = self.ell(t) - mu[-1]
    s = 1/abs(L[-1,-1])
    p = cdf(m/s)
    return self.k1*(m + self.l1)*(1-p) + self.k2*(m - self.l2)*p + (self.k2-self.k1)*s*pdf(m/s)

  @partial(jit,static_argnums = (0,))
  def entropyProduction(self,mu,Sinv,t):
    L = np.linalg.cholesky(Sinv)
    dmu = bDiff(mu)
    dmuLambda = self.ell(t) - mu[-1]
    D1 = np.eye(self.nMasses) - np.diag(np.ones(self.nMasses -1),-1)
    stds = np.sqrt(np.diag(D1 @ choSolve(L,D1.T) ))
    stdLambda = abs(1/L[-1,-1])
  
    mMu = (self.k1*(dmu + self.l1)*cdf(-dmu/stds) +
            self.k2*(dmu - self.l2)*cdf(dmu/stds) +
            (self.k2 - self.k1)*stds*pdf(dmu/stds) )
    mLambda = (self.k1*(dmuLambda + self.l1)*cdf(-dmuLambda/stdLambda) +
            self.k2*(dmuLambda - self.l2)*cdf(dmuLambda/stdLambda) +
            (self.k2 - self.k1)*stdLambda*pdf(dmuLambda/stdLambda) )

    muDot = fDiff(mMu)/self.eta
    muDot = muDot.at[-1].set(muDot[-1] + mLambda/self.eta)
   
    wL = (self.k1*cdf(-dmu/stds) + self.k2*cdf(dmu/stds)
           -(self.k1*self.l1 + self.k2*self.l2)*pdf(dmu/stds)/stds )
    wL = wL/self.eta
    wLambda = (self.k1*cdf(-dmuLambda/stdLambda)
                +self.k2*cdf(dmuLambda/stdLambda)
                -(self.k1*self.l1 + self.k2*self.l2)*pdf(dmuLambda/stdLambda)/stdLambda )
    wLambda = wLambda/self.eta
    D2w = -(D1.T*wL) @ D1
    D2w = D2w.at[-1,-1].set(D2w[-1,-1] - wLambda)
    S = np.linalg.inv(Sinv)
    dAdS = self.eta/2*(S@ D2w @ S + self.D*S)
    SinvDot = -2*((D2w @ Sinv + Sinv @ D2w)/2 + self.D *Sinv@Sinv)
    return self.eta*(muDot @ muDot) - np.trace(dAdS @ SinvDot.T)

  def phase(self,mu,Sinv,t):
    L = np.linalg.cholesky(Sinv)
    D1 = np.eye(len(mu)) - np.diag(np.ones(len(mu)-1),-1)
    stds = np.sqrt(np.diag(D1 @ choSolve(L, D1.T)))
    dmu = D1 @ mu
    dmuL = self.ell(t) - mu[-1]
    sL = abs(1/L[-1,-1])
    ps = np.zeros(mu.shape[0]+1)
    ps = ps.at[:-1].set(cdf(dmu/stds))
    ps = ps.at[-1].set(cdf(dmuL/sL))
    return ps

  def phases(self,mus,Sinvs,ts):
    return vmap(self.phase)(mus,Sinvs,ts)

  def getFrontLoc(self,mus,Sinvs,ts):
    ps = self.phases(mus,Sinvs,ts)
    tphase = []
    for p in ps.T:
      i = bl(p,.5)
      tphase.append(ts[i])
    return np.array(tphase)
    

