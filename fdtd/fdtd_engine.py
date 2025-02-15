import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as patches
import fdtd

fdtd.setup_log()
## CONSTANTS ####
c = 299792458 ## speed of light 
eta_0 = 377 ## impedance of free space
eps_0 = 8.85418782e-12
miu_0 = 1.25663706e-6


#### MAIN FDTD CLASSS ###########
class FDTD_GRID:
    ## Constructor
    '''
    args should be a dictionary with the following keys:
    - dim -> dimension of sim (2D/3D)
    - sampling_freq -> Sampling frequency of discrete domain
    - courant_num -> courant number
    - max_time -> maximum simulation time
    - max_x -> max value of x 
    - PML -> T/F if PML boundary required
    - stride -> how many time steps to print
    - max_y -> max value of y [Only required if 2D simulation]
    - 2d_mode -> mode of 2D wave TMZ or TEZ [Only required for 2D simulation]
    '''
    def __init__(self, args):
        
        ## logging
        self.logger = logging.getLogger("FDTD")

        ## Init Parameters
        self.fs = args["sampling_freq"]
        self.delta_t = (1/self.fs)
        self.Sc = args["courant_num"]
        self.dim = args["dim"]
        self.t_max = args["max_time"]
        self.x_max = args["max_x"]
        self.delta_x = (c*self.delta_t)/self.Sc
        self.state = "INIT" ## INIT -> MAT -> PML -> COEF -> READY
        
        # PML Parameters
        self.HAS_PML = args["PML"]
        self.X_PML_SIZE = 0
        self.Y_PML_SIZE = None

        ## Source Parameter
        self.src_added = False
        self.src_x_indx_strt = None
        self.src_x_indx_end = None
        self.src_func = None

        ## Redundant parameters
        self.total_x = round(self.x_max/self.delta_x)
        self.total_t = round(self.t_max/self.delta_t)
        
        ## Materials 
        self.epsilon = None
        self.miu = None
        self.sigma_e = None
        self.sigma_m = None
        self.eta = None
        self.HAS_MAT = False
        self.mat_idx = []

        ## Fields
        self.E = None
        self.H = None
        self.update_eq = None ## Function used to update equations
        self.time_stepping_fn = None
        
        ## coefficients
        self.coef = None

        ## Check dimensions ok
        if(self.dim != '1D' and self.dim != '2D'):
            self.logger.critical(f"Dimension {self.dim} not supported!")
            exit(-1)
            
        if(self.dim == '2D'):
            self.logger.info(f"Courant Number: {self.Sc} -- In 2D best is: {1/np.sqrt(2)}")
            self.y_max = args["max_y"]
            self.delta_y = self.delta_x ## FIXME: delta_y not calculated // add support for not nxn grids
            self.total_y = round(self.y_max/self.delta_y) 
            self.Y_PML_SIZE = 0
            self.mode_2d = args["2d_mode"]
            self.src_y_indx_strt = None
            self.src_y_indx_end = None
            
           
        else:
            self.logger.info(f"Courant Number: {self.Sc} -- In 1D best is: 1")
        
        ## Check if PML needs to be set
        if(self.HAS_PML):
            self.set_PML()
            logging.info(f"Size of PML: {self.X_PML_SIZE} nodes on each end")
        
        ## initialize materials
        self.init_materials()

        ## init fields
        self.init_fields()

        ## Plot stuff
        self.animation_obj = None
        self.fig = None
        self.ax = None
        self.save_ani = False
        self.ani_name = None
        self.stride = args['stride']
        ## Initialization Finished
        self.logger.info(f"delta_t: {self.delta_t} -- #steps: {self.total_t}")
        self.logger.info(f"delta_x: {self.delta_x} -- #steps: {self.total_x}")
        self.logger.info("Init Routine Finished!")
        ## change to MAT state (ready to add materials to grid)
        self.state = 'MAT'
    
    def init_materials(self):
        if(self.dim == '1D'):
            vec_size = self.total_x + 2*self.X_PML_SIZE
            self.epsilon = np.ones(vec_size) * eps_0
            self.miu = np.ones(vec_size) * miu_0
            self.sigma_e = np.zeros(vec_size)
            self.sigma_m = np.zeros(vec_size)
        else:
            vec_size_x = self.total_x + 2*self.X_PML_SIZE
            vec_size_y = self.total_y +  2*self.Y_PML_SIZE
            self.epsilon = np.ones((vec_size_x, vec_size_y)) * eps_0
            self.miu = np.ones((vec_size_x, vec_size_y)) * miu_0

            ## for split PML we need two versions
            if(self.HAS_PML):
                self.sigma_e={'x': np.zeros((vec_size_x, vec_size_y)), ## for waves propagating in x direction
                              'y':  np.zeros((vec_size_x, vec_size_y))} ## for waves propagating in y direction
                self.sigma_m={'x': np.zeros((vec_size_x, vec_size_y)), ## for waves propagating in x direction
                              'y':  np.zeros((vec_size_x, vec_size_y))} ## for waves propagating in y direction
            ## No PML we are good with one
            else:
                self.sigma_e = np.zeros((vec_size_x, vec_size_y))
                self.sigma_m = np.zeros((vec_size_x, vec_size_y))

    '''
    Transverse Magnetic Mode Assumptions:
        - Hx(x,y) and Hy(x,y) != 0
        - Ez(x,y) != 0
        - Hz = 0
        - Ex and Ey = 0
    The selection of xy for propagation is arbitrary!
    '''
    def set_tmz_fields(self):

        ## for split PML Ez has to be split in two parts a x traveling component and a y traveling component Ez = Ez_x + Ez_y
        if(self.HAS_PML):
            self.E = {'Ez': np.zeros((self.total_x + 2*self.X_PML_SIZE, self.total_x + 2*self.Y_PML_SIZE)),
                       'Ezx': np.zeros((self.total_x + 2*self.X_PML_SIZE, self.total_x + 2*self.Y_PML_SIZE)),
                       'Ezy': np.zeros((self.total_x + 2*self.X_PML_SIZE, self.total_x + 2*self.Y_PML_SIZE))}
            self.H = {'Hx': np.zeros((self.total_x + 2*self.X_PML_SIZE, self.total_x + 2*self.Y_PML_SIZE)),
                       'Hy': np.zeros((self.total_x + 2*self.X_PML_SIZE, self.total_x + 2*self.Y_PML_SIZE))}
        else:
            self.E = {'Ez':np.zeros((self.total_x + 2*self.X_PML_SIZE,self.total_x + 2*self.Y_PML_SIZE))}
            self.H = {'Hx': np.zeros((self.total_x + 2*self.X_PML_SIZE, self.total_x + 2*self.Y_PML_SIZE)),
                       'Hy': np.zeros((self.total_x + 2*self.X_PML_SIZE, self.total_x + 2*self.Y_PML_SIZE))}
        
        
        
    '''
     1D wave Assumptions
     Ez(x) != 0 // Does not change with y or z
     Hy(x) != 0 // Does not change with y or z
     Ex and Ey = 0
     Hx and Hz = 0
     Propagates in x direction 
     This is linear polarization 
     The selection of coordinates x,y, and z is arbitrary
    '''
    def init_fields(self):
        if(self.dim == '1D'):
            self.E = np.zeros(self.total_x + 2*self.X_PML_SIZE)
            self.H = np.zeros(self.total_x + 2*self.X_PML_SIZE)
            self.update_eq = self.update_1d
        else:
            if(self.mode_2d == "TMZ"):
                self.set_tmz_fields()
                self.update_eq = self.update_2d
            else:
                self.logger.critical(f"Only TMZ supported")
                exit(-1)
                
    def set_PML(self):
        ## PML size set to 5% of dimension space on each side 
       
        
        if (self.dim == '2D'):
            '''
            FIXME: Only nxn simulation space supported rn
            '''
            self.X_PML_SIZE = int(0.1*round(self.total_x))
            self.Y_PML_SIZE = self.X_PML_SIZE
        else:
            self.X_PML_SIZE = int(0.05*round(self.total_x))
          
    
    '''
    This function should be AFTER materials are added to the simulation space
    '''
    def PML_apply(self, initial_guess):
        if(not self.HAS_PML):
            self.logger.critical("Trying to applied PML when PML field was set to false during grid creation")
            exit(-1)

        ## check PML is calculated in correct state
        if(self.state != 'MAT'):
            self.logger.critical(f"You must be in MAT state to calculate PML. You're in {self.state}")
            exit(-1)

        if (self.dim == '1D'):
            ## Some initial guess for sigma_e 
            ## If sigma is too large causes a lot reflection from sim/PML boundary
            ## If sigma is too small causes a lot of reflection from PML/end boundary -- energy that eventually returns to the sim
            self.sigma_e[:self.X_PML_SIZE] = initial_guess / (self.epsilon[self.X_PML_SIZE + 1]/eps_0) ## scaled by adjacent eps_r
            self.sigma_e[-self.X_PML_SIZE:] = initial_guess / (self.epsilon[-self.X_PML_SIZE - 1]/eps_0)
            
            ## sigma_m is calculated to match impedance
            self.sigma_m[:self.X_PML_SIZE] =  np.multiply(self.sigma_e[:self.X_PML_SIZE], np.divide(self.miu[:self.X_PML_SIZE],self.epsilon[:self.X_PML_SIZE]))
            self.sigma_m[-self.X_PML_SIZE:] =  np.multiply(self.sigma_e[-self.X_PML_SIZE:] , np.divide(self.miu[-self.X_PML_SIZE:],self.epsilon[-self.X_PML_SIZE:]))

            ## matching epsilons and mius
            self.epsilon[:self.X_PML_SIZE] = self.epsilon[self.X_PML_SIZE + 1]
            self.epsilon[-self.X_PML_SIZE:] = self.epsilon[-self.X_PML_SIZE - 1]

            self.miu[:self.X_PML_SIZE] = self.miu[self.X_PML_SIZE + 1]
            self.miu[-self.X_PML_SIZE:] = self.miu[-self.X_PML_SIZE - 1]
            
        else:
            ## sigma ex
            self.sigma_e['x'][:,:self.X_PML_SIZE] = initial_guess / (self.epsilon[:,self.X_PML_SIZE+1]/eps_0).reshape(len(self.epsilon[:,self.X_PML_SIZE+1]), 1)
            self.sigma_e['x'][:,-self.X_PML_SIZE:] =  initial_guess / (self.epsilon[:,-self.X_PML_SIZE -1]/eps_0).reshape(len(self.epsilon[:,-self.X_PML_SIZE -1]), 1)

            ## sigma ey
            self.sigma_e['y'][:self.Y_PML_SIZE,:] = initial_guess / (self.epsilon[self.Y_PML_SIZE+1,:]/eps_0).reshape(1, len(self.epsilon[:,self.Y_PML_SIZE+1]))
            self.sigma_e['y'][-self.X_PML_SIZE:,:] =  initial_guess / (self.epsilon[-self.Y_PML_SIZE -1,:]/eps_0).reshape(1, len(self.epsilon[:,-self.Y_PML_SIZE -1]))
            
            ## sigma m should match impedance
            self.sigma_m['x'][:,:self.X_PML_SIZE] = np.multiply(self.sigma_e['x'][:,:self.X_PML_SIZE], np.divide(self.miu[:,:self.X_PML_SIZE], self.epsilon[:,:self.X_PML_SIZE]))
            self.sigma_m['x'][:,-self.X_PML_SIZE:] = np.multiply(self.sigma_e['x'][:,-self.X_PML_SIZE:], np.divide(self.miu[:,-self.X_PML_SIZE:], self.epsilon[:,-self.X_PML_SIZE:]))

            self.sigma_m['y'][:self.Y_PML_SIZE,:] = np.multiply(self.sigma_e['y'][:self.Y_PML_SIZE,:], np.divide(self.miu[:self.Y_PML_SIZE,:], self.epsilon[:self.Y_PML_SIZE,:]))
            self.sigma_m['y'][-self.X_PML_SIZE:,:] = np.multiply(self.sigma_e['y'][-self.X_PML_SIZE:,:], np.divide(self.miu[-self.X_PML_SIZE:,:], self.epsilon[-self.X_PML_SIZE:,:]))
            
            for mat_prop in (self.epsilon, self.miu):
                mat_prop[:,:self.X_PML_SIZE] = mat_prop[:,self.X_PML_SIZE+1].reshape(len(mat_prop[:,self.X_PML_SIZE+1]), 1)
                mat_prop[:,-self.X_PML_SIZE:] = mat_prop[:,-self.X_PML_SIZE -1].reshape(len(mat_prop[:,-self.X_PML_SIZE -1]), 1)
                mat_prop[:self.Y_PML_SIZE,:] =  mat_prop[self.Y_PML_SIZE+1,:].reshape(1, len(mat_prop[:,self.Y_PML_SIZE+1]))
                mat_prop[-self.X_PML_SIZE:,:] = mat_prop[-self.Y_PML_SIZE -1,:].reshape(1, len(self.epsilon[:,-self.Y_PML_SIZE -1]))

        self.state = 'PML'
        

    '''
    args should be a dictionary with the following keys
    - x_start -> distance in m where the material starts (x coordinate)
    - x_end -> distance in m where the material ends (x coordinate)
    - eps_r -> relative permitivity 
    - miu_r -> relative permeability 
    - sigma_e -> assuming the block has uniform conductivity 
    - y_start -> distance in m where the material starts (y coordinate) [Only 2D]
    - y_end -> distance in m where the material ends (y coordinate)[Only 2D]
    '''
    def add_material(self,args):

        ## check that adding materials is not being called during incorrect state (i.e. after PML calcualtion')
        if(self.state != 'MAT'):
            self.logger.critical("Materials can only be added during MAT state...Now you're in: {self.state}")
            exit(-1)
       
        ### Calculate the index for x position
        idx_start = int(args['x_start']/self.delta_x) + self.X_PML_SIZE
        idx_end = int(args['x_end']/self.delta_x) + self.X_PML_SIZE
        
        if (self.dim == '1D'):
            
            if(idx_start==idx_end):
                self.logger.critical("Point Material not supported")
                exit(-1)

            self.epsilon[idx_start:idx_end] = self.epsilon[idx_start:idx_end] * args["eps_r"]
            self.miu[idx_start:idx_end] = self.miu[idx_start:idx_end] * args["miu_r"]
            self.sigma_e[idx_start:idx_end] = args["sigma_e"]
            self.logger.info(f"Material with er={args["eps_r"]}, ur={args["miu_r"]}, and sig={args["sigma_e"]} added to from node {idx_start} to {idx_end}")
            
            mat_col = 'red' if 'color' not in args else args['color']
            self.mat_idx.append((idx_start, idx_end, mat_col))
        
        else:
            idx_start_y = int(args['y_start']/self.delta_x) + self.Y_PML_SIZE
            idx_end_y = int(args['y_end']/self.delta_x) + self.Y_PML_SIZE
            if(idx_start == idx_end and idx_start_y==idx_end_y):
                self.logger.critical("Single Point Material not supported")
                exit(-1)
            self.epsilon[idx_start_y:idx_end_y,idx_start:idx_end] = self.epsilon[idx_start_y:idx_end_y, idx_start:idx_end]*args["eps_r"]
            self.miu[idx_start_y:idx_end_y, idx_start:idx_end] = self.miu[idx_start_y:idx_end_y, idx_start:idx_end]  *  args["miu_r"]
            self.sigma_e[idx_start_y:idx_end_y,idx_start:idx_end] = args["sigma_e"]
            self.logger.info(f"Material with er={args["eps_r"]}, ur={args["miu_r"]}, and sig={args["sigma_e"]} added to from node ({idx_start},{idx_end}) to node ({idx_start_y, idx_end_y})")
            
            mat_col = 'red' if 'color' not in args else args['color']
         
            ## Material idex for 2D must be the actual (float) position with respect to axis not integer index
            self.mat_idx.append((args['x_start'], args['x_end'], args['y_start'], args['y_end'], mat_col))

        self.HAS_MAT = True
       
        
    '''
    A dictionary with the following keys is expected:
        - src_func: a function that takes a single argument t and returns a np.array of the correct size
        - x_start =  for the source start location of source in m
        - x_end = x for the end location of the source in m
        - y_start = y for the source start location of source in m [required for 2D only]
        - y_end = y for the end location of the source in m [required for 2D only]

        NOTE: If point source used set start and end to same point
    '''
    def add_src(self, args):
        self.src_x_indx_strt = int(args["x_start"]/self.delta_x) + self.X_PML_SIZE
        self.src_x_indx_end = max(self.src_x_indx_strt + 1, int(args["x_end"]/self.delta_x) + self.X_PML_SIZE) ## this deals with point src
        
        
        if(self.dim == '2D'):
            ## FIXME: this code might need refactoring if delta_x != delta_y
            self.src_y_indx_strt = int(args["y_start"]/self.delta_x) + self.Y_PML_SIZE
            self.src_y_indx_end = max(self.src_y_indx_strt+1, int(args["y_end"]/self.delta_x) + self.Y_PML_SIZE) ## this deals with point src
            self.logger.info(f"Source located at nodes [{self.src_x_indx_strt}: {self.src_x_indx_end}, {self.src_y_indx_strt}:{self.src_y_indx_end}]")
        
        else:
            self.logger.info(f"Source located at nodes [{self.src_x_indx_strt}: {self.src_x_indx_end}]")

        self.src_func = args["src_func"]
        self.src_added = True
        

    def tmz_coeff(self):
        '''
        Nomenclature (same as Schneider):
        C -> Coefficient
        letter , letter -> field being updated and components
        letter -> field being multiplied by this constant
        i.e., Chxh -> constant that multiplies H field in the Hx update equation
        '''
        if(self.HAS_PML):
            self.coef = {'Chxh': np.divide(1-np.divide(self.sigma_m['y']*self.delta_t, 2*self.miu), 1+np.divide(self.sigma_m['y']*self.delta_t, 2*self.miu)),
                         'Chxe': np.multiply(1/(1+np.divide(self.sigma_m['y']*self.delta_t, 2*self.miu)), (self.delta_t/self.delta_x)/(self.miu)),
                         'Chyh': np.divide(1-np.divide(self.sigma_m['x']*self.delta_t, 2*self.miu), 1+np.divide(self.sigma_m['x']*self.delta_t, 2*self.miu)),
                         'Chye': np.multiply(1/(1+np.divide(self.sigma_m['x']*self.delta_t, 2*self.miu)), (self.delta_t/self.delta_x)/self.miu),
                         'Cezxe': np.divide(1-np.divide(self.delta_t*self.sigma_e['x'], 2*self.epsilon), 1+np.divide(self.delta_t*self.sigma_e['x'],2*self.epsilon)),
                         'Cezxh': np.multiply(1/(1+np.divide(self.sigma_e['x']*self.delta_t, 2*self.epsilon)), (self.delta_t/self.delta_x)/self.epsilon),
                         'Cezye': np.divide(1-np.divide(self.delta_t*self.sigma_e['y'], 2*self.epsilon), 1+np.divide(self.delta_t*self.sigma_e['y'],2*self.epsilon)),
                         'Cezyh': np.multiply(1/(1+np.divide(self.sigma_e['y']*self.delta_t, 2*self.epsilon)), (self.delta_t/self.delta_x)/self.epsilon)
                        }
        else:
            self.coef = {'Chxh': np.divide(1-np.divide(self.sigma_m*self.delta_t, 2*self.miu), 1+np.divide(self.sigma_m*self.delta_t, 2*self.miu)),
                      'Chxe': np.multiply(1/(1+np.divide(self.sigma_m*self.delta_t, 2*self.miu)), (self.delta_t/self.delta_x)/(self.miu)),
                      'Chyh': np.divide(1-np.divide(self.sigma_m*self.delta_t, 2*self.miu), 1+np.divide(self.sigma_m*self.delta_t, 2*self.miu)),
                      'Chye': np.multiply(1/(1+np.divide(self.sigma_m*self.delta_t, 2*self.miu)), (self.delta_t/self.delta_x)/(self.miu)),
                      'Ceze': np.divide(1-np.divide(self.delta_t*self.sigma_e, 2*self.epsilon), 1+np.divide(self.delta_t*self.sigma_e,2*self.epsilon)),
                      'Cezh': np.multiply(1/(1+np.divide(self.sigma_e*self.delta_t, 2*self.epsilon)), (self.delta_t/self.delta_x)/self.epsilon)
                       }
        print(self.coef['Chxh'].shape)
    '''
    This should be called after materials have been added and PML has been applied
    '''
    def coeff_calculation(self):
        ## check state
        if( self.HAS_PML == True and self.state != 'PML'):
            self.logger.critical(f"To calculate coef you should be in PML state. You're in {self.state}")
            exit(-1)
        ## if grid does not have PML then materials should have been already added
        elif(self.HAS_PML == False and self.state != 'MAT'):
            self.logger.critical(f"To calculate coef you should be in MAT state. You're in {self.state}")
            exit(-1)


        if(self.dim == '1D'):
           '''
           Nomenclature ():
           C -> Coefficient
           # -> Position in equation
           letter -> filed being updated
           i.e., C1e -> first constant in the electric field equation
          
           '''
           self.coef= { 'C1e': np.divide(1-np.divide(self.sigma_e*self.delta_t,(2*self.epsilon)), 1+np.divide(self.sigma_e*self.delta_t,(2*self.epsilon))),
                        'C2e': np.divide(self.delta_t/(self.epsilon*self.delta_x), 1 + np.divide(self.sigma_e*self.delta_t, (2*self.epsilon))),
                        'C1h': np.divide(1-np.divide(self.sigma_m*self.delta_t,(2*self.miu)), 1+np.divide(self.sigma_m*self.delta_t,(2*self.miu))),
                        'C2h': np.divide(self.delta_t/(self.miu*self.delta_x), 1 + np.divide(self.sigma_m*self.delta_t, (2*self.miu)))
                      }
        else:
            if(self.mode_2d == 'TMZ'):
                self.tmz_coeff()
        
        self.state = 'READY'
        self.logger.info("Finished Coefficients Calculation!")

    def time_stepping_1d(self, t,src_args):
     
        ## update magnetic field
        ## H(x+1/2,t+1/2) = H(x+1/2,t+1/2) + [delta_t/(miu*delta_x)] (E(x+1,t)-E(x,t))
        self.H[:-1] = np.multiply(self.coef['C1h'][:-1],self.H[:-1]) + np.multiply(self.coef['C2h'][:-1],(self.E[1:] - self.E[:-1]))
        self.H[-1] = 0 ## set last node to zero 

        ## update E1
        ##E(x,t+1) = E(x,t) + [delta_t/(epsilon*delta_x)](H(x+1/2, t+1/2) - H(x-1/2, t+1/2))
        self.E[1:] = np.multiply(self.coef['C1e'][1:],self.E[1:]) + np.multiply(self.coef['C2e'][1:],(self.H[1:] - self.H[:-1]))
        self.E[0] = 0 ## set first node to zero

        ## add source term
        self.E[self.src_x_indx_strt:self.src_x_indx_end] = self.E[self.src_x_indx_strt:self.src_x_indx_end] + (self.delta_t/self.epsilon[self.src_x_indx_strt:self.src_x_indx_end])*self.src_func(t, **src_args)
    


    ########################################### 2D Time Stepping ####################################################

    '''
    With and without PML H updates equations are the same so this function is shared by both
    '''
    def tmz_update_h(self):
        ## from Schneider book page 188 (code) -- Equation 8.10 in page 186
        self.H['Hx'][:-1,:] =  np.multiply(self.coef['Chxh'][:-1,:], self.H['Hx'][:-1,:]) - np.multiply(self.coef['Chxe'][:-1, :],(self.E['Ez'][1:,:] - self.E['Ez'][:-1,:]))
        ## set last row (y=N) to zero
        self.H['Hx'][-1,:] = 0

        ## from Schneider book page 188 (code) -- Equation 8.11 in page 186
        self.H['Hy'][:,:-1] = np.multiply(self.coef['Chyh'][:,:-1], self.H['Hy'][:,:-1]) + np.multiply(self.coef['Chye'][:,:-1], (self.E['Ez'][:,1:] - self.E['Ez'][:,:-1]))
        ## set last column (x=M) to zero
        self.H['Hy'][:,-1] = 0
    
    def tmz_update_e_pml(self, t, src_args):
        self.E['Ezx'][:,1:] = np.multiply(self.coef['Cezxe'][:,1:], self.E['Ezx'][:,1:]) + np.multiply(self.coef['Cezxh'][:,1:], (self.H['Hy'][:,1:] - self.H['Hy'][:,:-1]))
        
        ## zero (first) column x=0 
        self.E['Ezx'][:,0] = 0


        self.E['Ezy'][1:,:] = np.multiply(self.coef['Cezye'][1:,:], self.E['Ezy'][1:,:]) - np.multiply(self.coef['Cezyh'][1:,:], (self.H['Hx'][1:,:]-self.H['Hx'][:-1,:]))
        ## zero (firs) row (Y = 0)
        self.E['Ezy'][0,:] = 0
    
        ## update Ez = Ezx + Ezy
        self.E['Ez'] = self.E['Ezx'] + self.E['Ezy']

    def tmz_update_e(self):
        ## from Schneider book page 188 (code) -- Equation 8.12 in page 188
        self.E['Ez'][1:,1:] = np.multiply(self.coef['Ceze'][1:,1:], self.E['Ez'][1:,1:]) + np.multiply(self.coef['Cezh'][1:,1:], ((self.H['Hy'][1:,1:] - self.H['Hy'][1:,:-1]) - (self.H['Hx'][1:,1:] - self.H['Hx'][:-1,1:])))
        ## set first row (Y=0) to zero
        ## set first column (X=0) to zero
        self.E['Ez'][0,:] = 0
        self.E['Ez'][:,0] = 0
        
    def time_stepping_2d_tmz(self, t, src_args):
       
        self.tmz_update_h()

        if(self.HAS_PML):
            self.tmz_update_e_pml(t, src_args)
        else:
            self.tmz_update_e()

        ## add source term
        self.E['Ez'][self.src_y_indx_strt:self.src_y_indx_end, self.src_x_indx_strt:self.src_x_indx_end] = self.E['Ez'][self.src_y_indx_strt:self.src_y_indx_end, self.src_x_indx_strt:self.src_x_indx_end] + (self.delta_t/self.epsilon[self.src_y_indx_strt:self.src_y_indx_end, self.src_x_indx_strt:self.src_x_indx_end])*self.src_func(t, **src_args)
        

    def update_2d(self, t, src_args):
        self.time_stepping_2d_tmz(t, src_args)
        if(int(t/self.delta_t) % self.stride == 0):
            self.animation_obj['e_field'].set_array(self.E['Ez'])
            self.animation_obj['hx_field'].set_array(np.multiply(self.eta,self.H['Hx']))
            self.animation_obj['hy_field'].set_array(np.multiply(self.eta,self.H['Hy']))
            self.animation_obj['time_txt'].set_text(f"t = {t*1e12:.2f} ps")
        return [self.animation_obj['e_field'], self.animation_obj['hx_field'], self.animation_obj['hy_field'], self.animation_obj['time_txt'], *self.animation_obj['material_drawings']]

    def runner_2d(self, amp_range):
        self.animation_obj = {'e_field': None,
                              'hx_field': None,
                              'hy_none': None,
                              'time_txt': None,
                              'material_drawings':[]}
        
        self.fig, self.ax = plt.subplots(1,3, figsize=(15, 5))   ## axes are 1 row 3 cols  
        txt_ax =  self.fig.add_axes([0, 0, 1, 1], frameon=False) # Invisible axes over entire figure
        bound_box = [-(self.X_PML_SIZE * self.delta_x), self.x_max + (self.X_PML_SIZE * self.delta_x), -(self.Y_PML_SIZE * self.delta_y), self.y_max + (self.Y_PML_SIZE * self.delta_y)]
        self.animation_obj['time_txt'] = txt_ax.text(0.5, 0.9, '', transform=txt_ax.transAxes, fontsize=12, color='red')
        self.animation_obj['e_field'] = self.ax[0].imshow(self.E['Ez'], cmap='plasma', animated=True, vmin=amp_range[0], vmax=amp_range[1],  extent=bound_box)
        self.animation_obj['hx_field'] = self.ax[1].imshow(np.multiply(self.H['Hx'], self.eta), cmap='plasma', animated=True, vmin=amp_range[0], vmax=amp_range[1], extent=bound_box)
        self.animation_obj['hy_field'] = self.ax[2].imshow(np.multiply(self.H['Hy'], self.eta), cmap='plasma', animated=True, vmin=amp_range[0], vmax=amp_range[1], extent=bound_box)
        
        cbar = self.fig.colorbar( self.animation_obj['e_field'], ax=self.ax, orientation="vertical", fraction=0.012)
        cbar.set_label("Field Intensity", rotation=-90)

        if(self.HAS_MAT):
            ## iterate through all materials
            for i, mats in enumerate(self.mat_idx):
                ## Calculate Hight: 
                height = mats[3] - mats[2]
                width = mats[1] - mats[0]
                for j in range(3):
                    self.animation_obj['material_drawings'].append(patches.Rectangle((mats[0], mats[2]), width, height, linewidth=5, facecolor=mats[4], zorder=10, alpha=0.4))
                    self.ax[j].add_patch(self.animation_obj['material_drawings'][3*i+j])
        
        if(self.HAS_PML):
            ## Horizontal PMLs 
            height_h = self.Y_PML_SIZE * self.delta_y
            width_h = self.x_max + 2*(self.X_PML_SIZE * self.delta_x)

            ## Vertical PMLs 
            height_v = self.y_max + 2*(self.Y_PML_SIZE * self.delta_y)
            width_v = self.X_PML_SIZE * self.delta_x
            
            for i in range(3):
                print((self.y_max-self.Y_PML_SIZE)*self.delta_x)
                self.animation_obj['material_drawings'].append(patches.Rectangle((-self.X_PML_SIZE*self.delta_x,-self.Y_PML_SIZE*self.delta_y), width_h, height_h, linewidth=1, facecolor='gray', zorder=10,alpha=0.5))
                self.animation_obj['material_drawings'].append(patches.Rectangle((-self.X_PML_SIZE*self.delta_x,self.y_max), width_h, height_h, linewidth=1, facecolor='gray', zorder=10,alpha=0.5))
                self.animation_obj['material_drawings'].append(patches.Rectangle((-self.X_PML_SIZE*self.delta_x,-self.Y_PML_SIZE*self.delta_y), width_v, height_v, linewidth=1, facecolor='gray', zorder=10,alpha=0.5))
                self.animation_obj['material_drawings'].append(patches.Rectangle((self.x_max,-self.Y_PML_SIZE*self.delta_y), width_v, height_v, linewidth=1, facecolor='gray', zorder=10,alpha=0.5))
                for j in range(4):
                    self.ax[i].add_patch(self.animation_obj['material_drawings'][-j-1])
               
            
        self.fig.text(0.02, 0.5, 'Y [m]', ha='left', va='center', rotation='vertical') 
        self.fig.text(0.5, 0.02, 'X [mm]', ha='center', va='bottom')
        self.ax[0].set_title('E Field')
        self.ax[1].set_title(r'$H_x$')
        self.ax[2].set_title(r'$H_y$')
        self.fig.tight_layout(rect=[0.02, 0.02, 0.85, 0.90])  

    def update_1d(self, t, src_args):
        self.time_stepping_1d(t, src_args)
        if(int(t/self.delta_t) % self.stride == 0):
            self.animation_obj["e_field"].set_ydata(self.E)
            self.animation_obj["h_field"].set_ydata(np.multiply(self.eta,self.H))
            self.animation_obj["time_txt"].set_text(f't = {t*1e12:.2f} ps')
        return self.animation_obj["e_field"], self.animation_obj["h_field"], self.animation_obj["time_txt"]

    def runner_1d(self, amp_range):

        ## FIXME: need to figure out a way to get this ranges adaptable based on factors
        ## animation params
        self.animation_obj = { "e_field": None,
                                "h_field":None,
                                "time_txt":None
                                }
        
        self.fig, self.ax = plt.subplots() 
        y_range = amp_range

        ## grid start value 
        grid_start = -self.X_PML_SIZE * self.delta_x
        grid_end = self.x_max + self.X_PML_SIZE * self.delta_x
        x = np.linspace(grid_start, grid_end, num=self.total_x+(2*self.X_PML_SIZE))

        self.animation_obj["e_field"], = self.ax.plot(x, self.E, label="E field")
        self.animation_obj["h_field"], = self.ax.plot(x, np.multiply(self.eta,self.H), label="z*H field", linestyle='--')
        self.ax.set_ylim(y_range[0], y_range[1])
        self.animation_obj["time_txt"] = self.ax.text(0.5, 0.9, '', transform=self.ax.transAxes, fontsize=12, color='red')
        self.ax.legend()
        
        if(self.HAS_PML):
            self.ax.fill_between(x[:self.X_PML_SIZE], y_range[0], y_range[1], color='gray', alpha=0.5)
            self.ax.fill_between(x[-self.X_PML_SIZE:], y_range[0], y_range[1], color='gray', alpha=0.5)
        
        if(self.HAS_MAT):
            for mats in self.mat_idx:
                self.ax.fill_between(x[mats[0]:mats[1]], y_range[0], y_range[1], color=mats[2], alpha=0.5)
        
        self.ax.set_xlabel("Distance [m]")
        self.ax.set_ylabel("Amplitude")

    '''
    amp range is tuple specifying the range of the plot/ the expected amplitudes of the fields
    src args is arbitrary variable that the source function should handle
    '''
    def start_sim(self, amp_range, src_args, sim_name=None):
        if(self.state != 'READY'):
            self.logger.critical(f"STATE IS NO READY! --- CURR.STATE: {self.state}")
            exit(-1)
        
        if(not self.src_added):
            self.logger.warning(f"No source has been added")

        if(sim_name):
            self.save_ani = True
            self.ani_name = sim_name
        
        ## calculate impedance to have E and H field that are visible in same graph
        self.eta = np.divide(self.miu, self.epsilon)
        self.eta = np.sqrt(self.eta)

        if(self.dim == '1D'):
            self.runner_1d(amp_range)
        
        else:
            self.runner_2d(amp_range)
        
        
        ani = FuncAnimation(self.fig, self.update_eq, fargs=(src_args,), frames=np.arange(0, self.t_max + self.delta_t , self.delta_t), interval=50, blit=True, repeat=False)
        
        if self.save_ani:
            ani.save(self.ani_name, writer="imagemagick", fps=30)
            logging.info("Animation saved to " + self.ani_name) 
        else:
            plt.show()
    '''
    start sim is catered towards producing an animation it will be interesting to make a more generic 
    thing that allows user to sneak peak into E and H field. Maybe simulate()? obviously a user can just iterate by iteself using time step
    '''