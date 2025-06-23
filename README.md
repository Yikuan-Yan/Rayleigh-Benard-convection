This repository is designed for numerical simulation for Rayleigh-Benard convection:
"Uniformly and gently heat the bottom of a container containing a suspension of powder in oil (e.g. mica powder in silicon oil), cell-like structures may form. Explain and investigate this phenomenon."
The codes are builded with LBM(Lattice Bolzmann Method). The codes takes these parameters as independent parameters:
- Experimental Parameters: size of container, height of liquid layer, temperature difference
- Simulation Parameters: number of simulation grids, simulation time interval, LBM relaxation time
- Fluid Properties: Viscosity, Thermal diffusivity
- Fluid Dimensionless Number: Prandtl number, Rayleigh number

Brief Infomation on codes:
- RB.py: Do 2D Simulation, Produce 2D GIF
- RBcount.py: Do 2D Simulation, Produce 2D GIF, output number of convections in the terminal
- 3D_RB.py: Do 3D Simulation. Please note this code have bugs, not fully developed.

Additional Information:
- Especially, these codes are designed for IYPT 2025 Problem 10. Everyone can use them for free, but please star this repo.
- The codes are written on 27/03/2025-29/03/2025, with the help of ChatGPT o4-mini-high.
