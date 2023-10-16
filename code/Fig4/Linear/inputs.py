from jax import random
import jax.numpy as np
beta,eta = 1.,1.
nMasses = 8

k1 = 1.
l1 = 1.
k2 = 1.
l2 = 1.

u = 1/2
ell = lambda t: -(nMasses + 1)*l1*(1 - u*t)
key = random.PRNGKey(1)
