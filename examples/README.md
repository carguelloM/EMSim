# Usage
## FDTD Class

Need to write here

## Examples
```
python3 1d_fdtd_simple.py --mode [S|P] ## S for saving to gif and P for printing to screen
```
```
python3 1d_fdtd_two_materials.py --mode [S|P] ## S for saving to gif and P for printing to screen
```
# Examples

## 1D

1D plane wave simulation with **fixed** boundaries (fields are **zero**). There is single 30 GHz source (infinite sheet) at the  middle of the grid (x=50mm). Note that $\mathbf{H}$ is multiplied by the impedance of the medium ($\eta_0$) for visualization purposes ($\mathbf{|E|} = \mathbf{|H|}$).

**Command**:

`python3 1d_fdtd_simple.py --mode P`

**Output**:
![](sims/1D_fdtf_simple.gif)



1D plane wave simulation with **fixed** boundaries (fields are **zero**). There is single 30 GHz source (infinite sheet) at the  middle of the grid (x=50 mm). $x \lt 40$ mm is filled with a material that has $\epsilon_r$ = 4 while everything to the right is free space. Note that $\mathbf{H}$ is multiplied by the impedance of the medium ($\eta_0$ for $x \geq 40$  mm and $\eta_1$ for $x \lt 40$ mm) for visualization purposes ($\mathbf{|E|} = \mathbf{|H|}$).

**Command**:

`python3 1d_fdtd_dielectric.py --mode P`

**Output**:
![](sims/1d_fdtd_dielectric.gif)

1D plane wave simulation with PML layers at boundaries at the ends of the simulation space (shown in `grey`). There is single 30 GHz source (infinite sheet) at the  middle of the grid (x=50mm).Note that $\mathbf{H}$ is multiplied by the impedance of the medium ($\eta_0$) for visualization purposes ($\mathbf{|E|} = \mathbf{|H|}$).

**Command**:

`python3 1d_fdtd_pml.py --mode P`

**Output**:
![](sims/1d_fdtd_pml.gif)

1D plane wave simulation with PML layers at boundaries at the ends of the simulation space (shown in `grey`). There is single 30 GHz source (infinite sheet) at the  middle of the grid (x=50 mm). $x \lt 40$ mm is filled with a material that has $\epsilon_r$ = 4 while everything to the right is free space. Note that $\mathbf{H}$ is multiplied by the impedance of the medium ($\eta_0$ for $x \geq 40$  mm and $\eta_1$ for $x \lt 40$ mm) for visualization purposes ($\mathbf{|E|} = \mathbf{|H|}$).

**Command**:

`1d_fdtd_dielectric_pml.py --mode S`

**Output**:

![](sims/1d_fdtd_dielectric_pml.gif)

## 2D
2D TMZ wave simulation with **fixed** boundaries (fields are **zero**). There is 300 GHz point source at the  middle of the grid (x=4 mm, y=4mm). Note that $\mathbf{H}$ is multiplied by the impedance of the medium ($\eta_0$) for visualization purposes ($\mathbf{|E|} = \mathbf{|H|}$).

**Command**:

`python3 2d_fdtd_simple.py --mode P`

**Output**:

![](sims/2d_fdtd_simple.gif)

2D TMZ wave simulation with **fixed** boundaries (fields are **zero**). There is 300 GHz point source at the  middle of the grid ($x=4$ mm, $y=4$ mm). The space between $x=5$ mm and $x=8$ mm is filled with water ($\epsilon_r$ = 1.7, $\sigma$ =10). Note that $\mathbf{H}$ is multiplied by the impedance of the medium ($\eta_0$ for $x \lt 5$  mm and $\eta_w$ for $x \geq 5$ mm) for visualization purposes ($\mathbf{|E|} = \mathbf{|H|}$).

**Command**:
`python3 2d_fdtd_water.py --mode P`

![](sims/2d_fdtd_water.gif)