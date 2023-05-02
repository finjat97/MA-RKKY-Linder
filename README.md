# MA-RKKY-Linder
Master's thesis about RKKY interaction in non-centrosymmetric superconductors supervised by Jacob Linder

## Bogoliubov-de Gennes (BdG) approach
main_RKKY.py allows to study different scenarios like increasing triplet pairing, increasing spin orbit coupling etc. for a chosen system.
The scenario is chosen within the python file, while the system parameters are given via the command line.
RKKY_diagonalization_ky.py constructs the system's hamiltonian and transforms the y-real space coordinate into k-space, and then diagonalizes the hamiltonian.
observables.py contains the functions to calculate different observables based on the eigenvalues, eigenvectors and coefficients from the diagonaized hamiltonian.
routines.py are all the functions called by main_RKKY.py and calculate all necessary data for the different scenarios.

groundstate.py and boundstates.py are scenarios specifically tailored to produce results comparable to the SWT approach (see next paragraph) and 
they use the same functions as main_RKKY.py

plot.py simply plots the data from main_RKKY.py in a specific way.
3d_arrow_plot.py plots the groundstate spin configurations from groundstate.py as 3 dimensional arrows over distance.

## Schrieffer-Wolff transformation (SWT)
ana_comp_final.py computes the spinstructure for a normal metal or superconductor with spin orbit coupling based on the anayltically found expression.
ana_spin.py plots the groundstate spin configuration based on ana_comp_final.py and plots it as 3 dimensional arrows over distance.
plot_ana_coeffi.py plots the different coefficients over distance and the y-axis limits can be adjusted.
ana_nm_sc_compute.py computes the spin structure coefficients for normal metals and superconductors (without spin orbit coupling).
