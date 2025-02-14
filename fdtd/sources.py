
import numpy as np
'''
Point sinusoidal source
'''
def pnt_sin_src(t, f_src):
   
    return np.sin(2*np.pi*f_src*t)