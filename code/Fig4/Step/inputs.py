from jax import random
import jax.numpy as np
beta,eta = 1.,1.
nMasses = 8

k1 = 1.
l1 = 1.
k2 = 1.
l2 = 1.


indic = lambda x,a,b: 1/2*(1 + np.sign(x-a))*1/2*(1 - np.sign(x-b))

amp = nMasses/2*l1
tS = 2.
tP0 = 2.5
tP1 = 7.5
tE = 8.
ell = lambda t: -(nMasses + 1)*l1 + (amp/(tP0 - tS)*(t-tS)*indic(t,tS,tP0)
                                  +amp*indic(t,tP0,tP1)
                                  +-amp/(tE - tP1)*(t-tE)*indic(t,tP1,tE) )





key = random.PRNGKey(5)
