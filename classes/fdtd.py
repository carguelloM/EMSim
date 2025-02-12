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

def time_stepping_2d(t, args):
    E = args["e_field"]
    Hx = args["h_fieldx"]
    Hy = args["h_fieldy"]
    Chxh = args["Chxh"]
    Chxe = args["Chxe"]
    Chyh = args["Chyh"]
    Chye = args["Chye"]
    Ceze = args["Ceze"]
    Cezh = args["Cezh"]
    idx_src_x = args["idx_src_x"]
    idx_src_y = args["idx_src_y"]
    print(E[idx_src_x+2, idx_src_y])
    ## from Schneider book page 188 (code) -- Equation 8.10 in page 186
    Hx[:,:-1] =  np.multiply(Chxh[:,:-1], Hx[:,:-1]) - np.multiply(Chxe[:, :-1],(E[:,1:] - E[:,:-1]))
    ## set last column to zero
    Hx[:,-1] = 0

    ## from Schneider book page 188 (code) -- Equation 8.11 in page 186
    Hy[:-1,:] = np.multiply(Chyh[:-1,:], Hy[:-1,:]) + np.multiply(Chye[:-1,:], (E[1:,:] - E[:-1,:]))
    ## set last row to zero
    Hy[-1,:] = 0

    ## from Schneider book page 188 (code) -- Equation 8.12 in page 188
    E[1:,1:] = np.multiply(Ceze[1:,1:], E[1:,1:]) + np.multiply(Cezh[1:,1:], ((Hy[1:,1:] - Hy[:-1,1:]) - (Hx[1:,1:] - Hx[1:,:-1])))
    ## set first row and column to zero
    E[1,:] = 0
    E[:,1] = 0

    ## add source term
    E[idx_src_x, idx_src_y] =E[idx_src_x, idx_src_y] + args["del_t"]/args["eps"][idx_src_x, idx_src_y]*np.sin(2*np.pi*args["f_src"]*t)
    
    return E, Hx, Hy
