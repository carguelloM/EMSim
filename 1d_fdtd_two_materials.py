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
t_max = 400e-12 ## 100 ps
delta_t = (1/f_sampling)
Sc = 1 ## Courant number
delta_x = (c*delta_t)/Sc 
ratio_deltas = delta_t/delta_x
e_r = 1
u_r = 1
PML_SIZE = int(0.05*round(x_max/delta_x) + 1) ## PML size is 5% of sim space

## Vectors
PML_START = -PML_SIZE*delta_x
PML_END = x_max + PML_SIZE*delta_x
x = np.arange(PML_START,PML_END, delta_x)
epsilon = np.ones(len(x)) * permiti_free_space * e_r
miu = np.ones(len(x)) * premea_freee_space * u_r
sigma_e = np.zeros(len(x))
sigma_m = np.zeros(len(x))
E = np.zeros(len(x))
H = np.zeros(len(x))

#### MATERIAL RELATED ######

## e_r = 4 for x < 40 mm
indx_40mm = int(40e-3/delta_x) + PML_SIZE
epsilon[:indx_40mm] = epsilon[:indx_40mm] * 4 
## er = 1 for >= 40 mm already

## PML RELATED ####
## Some initial guess for sigma_e 
## If sigma is too large causes a lot reflection from sim/PML boundary
## If sigma is too small causes a lot of reflection from PML/end boundary -- energy that eventually returns to the sim
sigma_e[:PML_SIZE] = 3 * 4 ## Scaled by e_r at the left 
sigma_e[-PML_SIZE:] = 3 * e_r ## scaled by e_r to the right

## sigma_m is caluclated to match impedance
sigma_m[:PML_SIZE] = np.multiply(sigma_e[:PML_SIZE], np.divide(miu[:PML_SIZE],epsilon[:PML_SIZE]))
sigma_m[-PML_SIZE:] = np.multiply(sigma_e[-PML_SIZE:] , np.divide(miu[-PML_SIZE:],epsilon[-PML_SIZE:]))

## ONE-TIME calculation
## Coefficient ONE update to electric field 1- sigma_e*delta_t/(2*epsilon) / 1+sigma_e*delta_t/(2*epsilon)
C1e = np.divide(1-np.divide(sigma_e*delta_t,(2*epsilon)), 1+np.divide(sigma_e*delta_t,(2*epsilon)))

## Coeffient TWO update to electric field delta_t/(epsilon*delta_x) / 1+sigma_e*delta_t/(2*epsilon)
C2e = np.divide(delta_t/(epsilon*delta_x), 1 + np.divide(sigma_e*delta_t, (2*epsilon))) 

## Coeffiecnet ONE update to magnetic field 1-sigma_m*delta_t/(2*miu) / 1+sigma_m*delta_t/(2*miu)
C1m = np.divide(1-np.divide(sigma_m*delta_t,(2*miu)), 1+np.divide(sigma_m*delta_t,(2*miu)))

## Coefficient TWO update to magnetic field delta_t/(miu*delta_x) / 1+sigma_m*delta_t/(2*miu)
C2m = np.divide(delta_t/(miu*delta_x), 1 + np.divide(sigma_m*delta_t, (2*miu)))

## source should be a 50 mm, need to get index
indx_src = round(50e-3/delta_x) + PML_SIZE

#### SCALING FACTOR FOR PLOT ####
## impedance vector for multiplication with H (for scaling)
z=np.divide(miu, epsilon)
z=np.sqrt(z)

## Running Logic
MODE = "P" ## "P" -> Plot "S" -> Save

#Set Plot
y_range = (-2e-2, 2e-2)
fig, ax = plt.subplots()
E_field_graph, = ax.plot(x*1e3,E, label="E field")
H_field_graph, = ax.plot(x*1e3,np.multiply(z,H), label="z*H field", linestyle='--')
ax.set_ylim(y_range[0], y_range[1])
time_text = ax.text(0.5, 0.9, '', transform=ax.transAxes, fontsize=12, color='red')
ax.legend()
ax.fill_between(x[:indx_40mm]*1e3, y_range[0], y_range[1], color='lightblue', alpha=0.5)
ax.fill_between(x[:PML_SIZE]*1e3, y_range[0], y_range[1], color='red', alpha=0.5)
ax.fill_between(x[-PML_SIZE:]*1e3, y_range[0], y_range[1], color='red', alpha=0.5)
ax.set_xlabel("Distance [mm]")
ax.set_ylabel("Amplitude")

def update(t):
    global E,H, indx_src, f_src, MODE
    args = {"e_field":E, 
            "h_field": H, 
            "C1e":C1e,
            "C2e":C2e,
            "C1m":C1m,
            "C2m":C2m,
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

    ## Animation Name
    ## Note this input must be acquired before simulation look
    ## otherwise frames get stacked (i guess because they arent process while waiting for input?)
    if MODE == "S":
        name = input("Name for simulation:")
        fn = "sims/"+ name + ".gif"

    ## setup logging
    classes.setup_log()
    log = logging.getLogger("Program")
    log.info("Starting 1D FDTD simulation")
    logging.info(f"delta_t: {delta_t} - #steps:{round(t_max/delta_t) + 1} \n delta_x: {delta_x} - #steps:{round(x_max/delta_x) + 1}")
    logging.info(f"Index of Source is:{indx_src}")

    ani = FuncAnimation(fig, update, frames=np.arange(0, t_max + delta_t , delta_t), interval=50, blit=True, repeat=False)

    if MODE=="S":
        ani.save(fn, writer="imagemagick", fps=30)
        logging.info("Animation saved to" + fn)
    else:
        plt.show()
    logging.info("Program Finished")