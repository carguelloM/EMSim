## 1D Finite Difference Time Domain (FDTD)
import numpy as np
import classes
import logging
import matplotlib.pyplot as plt
from classes.constants import *
from matplotlib.animation import FuncAnimation

## Parameters
f_sampling = 5e12 ## 100 THz
f_src = 30e9 ## 30 GHz 
x_max = 100e-3 ## 100 mm 
t_max = 100e-12 ## 100 ps
delta_t = (1/f_sampling)
Sc = 1 ## Courant number
delta_x = (c*delta_t)/Sc 
ratio_deltas = delta_t/delta_x
e_r = 1
u_r = 1

## Vectors
x = np.arange(0,x_max+delta_x, delta_x)
epsilon = np.ones(len(x)) * permiti_free_space * e_r
miu = np.ones(len(x)) * premea_freee_space * u_r
E = np.zeros(len(x))
H = np.zeros(len(x))



## ONE-TIME calculation
ratio_deltas_div_miu = ratio_deltas/miu
ratio_deltas_div_epsilon = ratio_deltas/epsilon

## source should be a 50 mm, need to get index
indx_src = int(50e-3/delta_x)
logging.info(f"Index of Source is:{indx_src}")

#Set Plot
fig, ax = plt.subplots()
E_field_graph, = ax.plot(x,E, label="E field")
H_field_graph, = ax.plot(x,imp_free_space*H, label="H*377 field", linestyle='--')
ax.set_ylim(-0.7, 0.7)
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=12, color='red')
ax.legend()

def time_stepping(t):
    global E, H, ratio_deltas_div_miu, ratio_deltas_div_epsilon, indx_src

    logging.info(f"Update at t:{t}")
    ## update magnetic field
    ## H(x+1/2,t+1/2) = H(x+1/2,t+1/2) + [delta_t/(miu*delta_x)] (E(x+1,t)-E(x,t))
    ## calculate E(x+1,t)
    ## left shift E and append zero at the end
    E_shift_left = np.concatenate((E[1:], np.zeros(1,dtype=E.dtype))) 
    H = H + np.multiply(ratio_deltas_div_miu, (E_shift_left - E))
    H[-1] = 0 ## set last node to zero 

    ## update E1
    ##E(x,t+1) = E(x,t) + [delta_t/(epsilon*delta_x)](H(x+1/2, t+1/2) - H(x-1/2, t+1/2))
    ## right shift H (just updated) and append zero at the beginning
    H_shift_right = np.concatenate((np.zeros(1,dtype=H.dtype), H[:-1]))
    E = E + np.multiply(ratio_deltas_div_epsilon,(H - H_shift_right))
    E[0] = 0 ## set first node to zero

    ## add source term
    E[indx_src] = E[indx_src] + np.sin(2*np.pi*f_src*t)
    E_field_graph.set_ydata(E)
    H_field_graph.set_ydata(imp_free_space*H)

    time_text.set_text(f't = {t*1e12:.2f} ps')
    return E_field_graph,H_field_graph, time_text

if __name__=="__main__":
    ## setup logging
    classes.setup_log()
    logging.info("Starting 1D FDTD simulation")
    logging.info(f"delta_t: {delta_t} - #steps:{round(t_max/delta_t) + 1} \n delta_x: {delta_x} - #steps:{round(x_max/delta_t) + 1}")


    ani = FuncAnimation(fig, time_stepping, frames=np.arange(0, t_max + delta_t , delta_t), interval=50, blit=True, repeat=False)
    ani.save('sims/1D_simple_no_end.gif', writer="imagemagick", fps=30)