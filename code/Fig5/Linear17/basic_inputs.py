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

