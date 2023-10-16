from jax import random
import jax.numpy as np
beta0,eta0 = 2000.,1.75
nMasses = 17

eta = eta0*6*(nMasses + 1)/(2*nMasses + 1)/nMasses
beta = beta0*(nMasses + 1)

k10 = 3.
l10 = .75
k20 = .75
l20 = 1.25






k1 = k10*(nMasses + 1)
l1 = l10/(nMasses + 1)
k2 = k20*(nMasses + 1)
l2 = l20/(nMasses + 1)

u = 10


dt = float(5e-6)
dtL,dtE = dt,dt
Tf = 2.



nStepsTot = int(Tf/dt) + 1

nEval_e = 5000
nStepE = int(nStepsTot/nEval_e) + 1
nRep = 1

nEval_l = 500
nStepL = int(nStepsTot/nEval_l) + 1





ell = lambda t: -l10*(1-u*t)
key = random.PRNGKey(10)
