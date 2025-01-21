## 1D Finite Difference Time Domain (FDTD)
import numpy
import logging
## Parameters
f = 2.4e9 ## 2.4 GHz
x_max = 100e-3 ## 100 mm 
t_max = 100e-12 ## 100 ps
S_c = 1 ## courant number = 1
delta_t = (1/f) * 3 ## 3f to satisfy Nyquist
## 


## Vectors
t = np.arange(0,t_max, delta_t)
epsilon = np.ones(len(x))
miu = np.ones(len(x))


