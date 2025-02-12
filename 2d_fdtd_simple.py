import numpy as np
import classes
import logging
from classes.constants import *
import matplotlib.pyplot as plt
from classes.fdtd import time_stepping_2d
import argparse

## Parameters
f_sampling = 5e12 ## 5 THz >> f_src/2
f_src = 500e9 ## 500 GHz
x_max = 8e-3 ## 8 mm
y_max = 8e-3 ## 8 mm
t_max = 10e-12 ## 10 ps
delta_t = (1/f_sampling) 
Sc = 1/np.sqrt(2) ## Best Courant number in 2D 
delta_x = (c*delta_t)/Sc
delta_y = delta_x

total_x = round(x_max/delta_x)
total_y = round(x_max/delta_y)

## Vectors
xy_plane = np.zeros((total_x, total_y))
epsilon = np.ones((total_x, total_y))
miu = np.ones((total_x, total_y))
sigma_e = np.zeros((total_x, total_y))
sigma_m = np.zeros((total_x, total_y))
E = np.zeros((total_x, total_y))
Hx = np.zeros((total_x, total_y))
Hy = np.zeros((total_x, total_y))


##### MATERIAL RELATED ###
epsilon = epsilon * permiti_free_space 
miu = miu * premea_freee_space

### PML RELATED (TODO) ####



#### ONE TIME CALCUALTIONS ### 
Chxh = np.divide(1-np.divide(sigma_m*delta_t, 2*miu), 1+np.divide(sigma_m*delta_t, 2*miu))
Chxe = np.multiply(1/(1+np.divide(sigma_m*delta_t, 2*miu)), (delta_t/delta_x)/(miu))
Chyh = np.divide(1-np.divide(sigma_m*delta_t, 2*miu), 1+np.divide(sigma_m*delta_t, 2*miu)) 
Chye = np.multiply(1/(1+np.divide(sigma_m*delta_t, 2*miu)), (delta_t/delta_x)/miu)
Ceze = np.divide(1-np.divide(delta_t*sigma_e, 2*epsilon), 1+np.divide(delta_t*sigma_e,2*epsilon))
Cezh = np.multiply(1/(1+np.divide(sigma_e*delta_t, 2*epsilon)), (delta_t/delta_x)/epsilon)

### SOURCE RELATED ##

## source should be at center of simulation
idx_src_x = int((x_max/2)/delta_x)
idx_src_y = int((y_max/2)/delta_y)

def update(t, im):
    global E,Hx, Hy, idx_src_x, idx_src_y
    args_2d = { "e_field": E,
                "h_fieldx": Hx,
                "h_fieldy":Hy,
                "Chxh": Chxh,
                "Chxe": Chxe, 
                "Chyh": Chyh,
                "Chye": Chye,
                "Ceze": Ceze,
                "Cezh": Cezh,
                "idx_src_x": idx_src_x,
                "idx_src_y": idx_src_y, 
                "f_src":f_src,
                "eps":epsilon,
                "del_t": delta_t
                } 
    E, Hx, Hy = time_stepping_2d(t, args_2d)
    
    im.set_data(E)
    fig.canvas.draw()
    plt.pause(0.1)

fig, ax = plt.subplots()
im = ax.imshow(E, vmin=-0.005, cmap='plasma', vmax=0.005)
plt.colorbar(im)
print(f" Delta X: {delta_x}")
print(f"Total T: {round(t_max/delta_t)} -- Delta t {delta_t}")
t = 0
while t < t_max:
    update(t,im)
    t  = t + delta_t
plt.show()
