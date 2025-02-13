## 1D Finite Difference Time Domain (FDTD)

import classes
import logging
from classes.fdtd import FDTD_GRID
from classes.fdtd import pnt_sin_src_1d
import argparse
   
'''
Recipe to create a simulation
1. Create grid
2. Add material
3. Apply PML
4. Calculate coefficients
5. Add Source
6. Run the simulation
'''
if __name__=="__main__":
    ## Argument parsing
    parser = argparse.ArgumentParser(description="1D Simulation EM Wave in Free Space.")
    parser.add_argument("--mode", type=str, help="S(Save) & P(Print)")
    args = parser.parse_args()

    animation_fn = None

    if(args.mode != 'S' and args.mode != 'P'):
        print("Error: Invalid Operation Mode")
        exit(-1)

    
    if args.mode == "S":
        name = input("Name for simulation:")
        animation_fn = "sims/"+ name + ".gif"
    
    ## 1. Create Grid
    grid_param = { 'dim': '1D',
                   'sampling_freq': 5e12, ## 5 THz
                   'courant_num':1,
                   'max_time': 400e-12, ## 100 ps
                   'max_x': 100e-3, ## 100 mm
                   'PML': True,
                   'stride': 1
                   }
    myGrid = FDTD_GRID(grid_param)

    ## 2. NO materials
    ## 3. Apply PML
    pml_init_guess = 3
    myGrid.PML_apply(pml_init_guess)
    ## 4. Calculate Coefficients
    myGrid.coeff_calculation()

    ## 5. Add source
    src_args = { "src_func":pnt_sin_src_1d,
                "x_start":50e-3, ## 50 mm/middle of the grid
                "x_end":50e-3, ## start=end for pnt src
                }
    myGrid.add_src(src_args)

    ## 6. Start Simulation
    f_src_args={"f_src":30e9}
    myGrid.start_sim((-2e-2, 2e-2), f_src_args, animation_fn)
  
    
    

    