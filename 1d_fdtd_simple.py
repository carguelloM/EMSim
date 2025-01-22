## 1D Finite Difference Time Domain (FDTD)
import numpy as np
import classes
import logging
import matplotlib.pyplot as plt
from classes.constants import *
from classes.fdtd import time_stepping_1d
from matplotlib.animation import FuncAnimation
import argparse

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


## impedance vector for multiplication with H (for scaling)
z=np.divide(miu, epsilon)
z=np.sqrt(z)

## ONE-TIME calculation
ratio_deltas_div_miu = ratio_deltas/miu
ratio_deltas_div_epsilon = ratio_deltas/epsilon

## source should be a 50 mm, need to get index
indx_src = int(50e-3/delta_x)
logging.info(f"Index of Source is:{indx_src}")

## Running Logic
MODE = "P" ## "P" -> Plot "S" -> Save

#Set Plot
y_range = (-2e-2, 2e-2)
fig, ax = plt.subplots()
E_field_graph, = ax.plot(x,E, label="E field")
H_field_graph, = ax.plot(x,np.multiply(z,H), label="z*H field", linestyle='--')
ax.set_ylim(y_range[0], y_range[1])
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=12, color='red')
ax.legend()

def update(t):
    global E,H, ratio_deltas_div_epsilon, ratio_deltas_div_miu, indx_src, f_src, MODE
    args = {"e_field":E, 
        "h_field": H, 
        "del_miu":ratio_deltas_div_miu, 
        "del_eps":ratio_deltas_div_epsilon, 
        "idx_j":indx_src,
        "f_src":f_src,
        "del_t":delta_t,
        "eps":epsilon}
    
    E,H = time_stepping_1d(t, args)
    E_field_graph.set_ydata(E)
    H_field_graph.set_ydata(np.multiply(z,H))
    time_text.set_text(f't = {t*1e12:.2f} ps')
    return E_field_graph,H_field_graph, time_text

if __name__=="__main__":
    ## Argument parsing
    parser = argparse.ArgumentParser(description="1D Simulation EM Wave in Free Space.")
    parser.add_argument("--mode", type=str, help="S(Save) & P(Print)")
    args = parser.parse_args()
    if(args.mode != 'S' and args.mode != 'P'):
        print("Error: Invalid Operation Mode")
        exit(-1)
    
    MODE = args.mode

    ## setup logging
    classes.setup_log()
    logging.info("Starting 1D FDTD simulation")
    logging.info(f"delta_t: {delta_t} - #steps:{round(t_max/delta_t) + 1} \n delta_x: {delta_x} - #steps:{round(x_max/delta_t) + 1}")

    

    ani = FuncAnimation(fig, update, frames=np.arange(0, t_max + delta_t , delta_t), interval=50, blit=True, repeat=False)

    if MODE=="S":
        ani.save('sims/1D_simple_no_end.gif', writer="imagemagick", fps=30)
    else:
        plt.show()