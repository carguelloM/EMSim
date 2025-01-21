## 1D Finite Difference Time Domain (FDTD)
import numpy as np
import classes
import logging
import matplotlib.pyplot as plt
from classes.constants import *


## Parameters
f = 5e9 ## 5 GHz
x_max = 100e-3 ## 100 mm 
t_max = 100e-12 ## 100 ps
delta_t = (3/f) ## 3f to satisfy Nyquist
Sc = 0.9 ## Courant number
delta_x = c*delta_t* Sc 


## Vectors
x = np.arange(0,x_max, delta_x)
epsilon = np.ones(len(x))
miu = np.ones(len(x))
E = np.zeros(len(x))
H = np.zeros(len(x))

if __name__=="__main__":
    ## setup logging
    classes.setup_log()
    logging.info("Starting 1D FDTD simulation")
    logging.info(f"delta_t: {delta_t}, delta_x: {delta_x}")


