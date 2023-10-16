from jax import random
import jax.numpy as np
beta,eta = 1.,1.
nMasses = 8

k1 = 1.
l1 = 1.
k2 = 1.
l2 = 1.


amp = nMasses/2 * l1
omega = 2*np.pi/5.
ell = lambda t: amp*np.sin(omega*t) - (nMasses + 1)*l1
key = random.PRNGKey(2)
