from jax import random
import jax.numpy as np
k,l,beta,eta = 1.,1.,1.,1.
nMasses = 8

a0 = np.sqrt(1/k/beta)
omega = 2*np.pi/3
ell = lambda t:  a0*(1/2*(np.sin(omega/2.4 * t) + 1.33))*np.sin(omega*t)
key = random.PRNGKey(3)
