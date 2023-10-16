from jax import random
import jax.numpy as np
beta,eta = 4.,1.
nMasses = 2

k1 = 1.
l1 = 1.
k2 = 8.
l2 = 2.

u = 1/2
ell = lambda t: -(nMasses + 1)*l1*(2 - u*t)
key = random.PRNGKey(2)
