import logging
import numpy as np

def time_stepping_1d(t, args):
    E = args["e_field"]
    H = args["h_field"]
    C1e = args["C1e"]
    C2e = args["C2e"]
    C1m = args["C1m"]
    C2m =args["C2m"]
    indx_src=args["idx_j"]
    
    ## update magnetic field
    ## H(x+1/2,t+1/2) = H(x+1/2,t+1/2) + [delta_t/(miu*delta_x)] (E(x+1,t)-E(x,t))
    ## calculate E(x+1,t)
    ## left shift E and append zero at the end
    E_shift_left = np.concatenate((E[1:], np.zeros(1,dtype=E.dtype))) 
    H = np.multiply(C1m,H) + np.multiply(C2m,(E_shift_left - E))
    H[-1] = 0 ## set last node to zero 

    ## update E1
    ##E(x,t+1) = E(x,t) + [delta_t/(epsilon*delta_x)](H(x+1/2, t+1/2) - H(x-1/2, t+1/2))
    ## right shift H (just updated) and append zero at the beginning
    H_shift_right = np.concatenate((np.zeros(1,dtype=H.dtype), H[:-1]))
    E = np.multiply(C1e,E) + np.multiply(C2e,(H - H_shift_right))
    E[0] = 0 ## set first node to zero

    ## add source term
    E[indx_src] = E[indx_src] + args["del_t"]/args["eps"][indx_src]*np.sin(2*np.pi*args["f_src"]*t)

    return E, H