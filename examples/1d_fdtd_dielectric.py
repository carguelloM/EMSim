## 1D Finite Difference Time Domain (FDTD)

import logging
from fdtd import fdtd_engine as sim, sources as src
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
    
    # classes.setup_log()
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
                   'max_time': 100e-12, ## 100 ps
                   'max_x': 100e-3, ## 100 mm
                   'PML': False, ## No PML
                   'stride': 1
                   }
    myGrid = sim.FDTD_GRID(grid_param)

    ## 2. Add material
    mat_args = { 'x_start': 0,
                'x_end': 40e-3,
                'eps_r': 4,
                'miu_r': 1,
                'sigma_e': 0, ## dielectric
                'color': 'green'
                }
    myGrid.add_material(mat_args)
    ## 3. NO PML

    ## 4. Calculate Coefficients
    myGrid.coeff_calculation()

    ## 5. Add source
    src_args = { "src_func":src.pnt_sin_src,
                "x_start":50e-3, ## 50 mm/middle of the grid
                "x_end":50e-3, ## start=end for pnt src
                }
    myGrid.add_src(src_args)

    ## 6. Start Simulation
    f_src_args={"f_src":30e9}
    myGrid.start_sim((-2e-2, 2e-2), f_src_args, animation_fn)
  
    
    

    