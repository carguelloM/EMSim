import classes
import logging
from classes.fdtd import FDTD_GRID
from classes.fdtd import pnt_sin_src
import argparse
import math
f_src = 500e9 ## 500 GHz



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
    
    classes.setup_log()
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
    grid_param = { 'dim': '2D',
                   'sampling_freq': 5e12, ## 5 THz
                   'courant_num':1/math.sqrt(2),
                   'max_time': 10e-12, ## 10 ps
                   'max_x': 8e-3, ## 8 mm
                   'max_y': 8e-3, ## 8 mm
                   'PML': False, ## No PML
                   '2d_mode': 'TMZ',
                   'stride': 1
                   }
    myGrid = FDTD_GRID(grid_param)
   
    ## 2. materials
    mat_args ={'x_start': 5e-3,
               'x_end': 8e-3,
               'y_start': 0,
               'y_end':8e-3,
               'eps_r': 1.7,
               'miu_r':1,
                'sigma_e':10,}
    myGrid.add_material(mat_args)
    ## 3. NO PML

    ## 4. Calculate Coefficients
    myGrid.coeff_calculation()
  
    ## 5. Add source
    src_args = { "src_func":pnt_sin_src,
                "x_start":4e-3, ## 4 mm/middle of the grid
                "x_end":4e-3, ## start=end for pnt src
                "y_start":4e-3, ## 4 mm/middle of the grid
                "y_end":4e-3, ## start=end for pnt src
                }
    myGrid.add_src(src_args)
  
    ## 6. Start Simulation
    f_src_args={"f_src":500e9}
    myGrid.start_sim((-0.005, 0.005), f_src_args, animation_fn)