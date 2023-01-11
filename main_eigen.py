import observables as o 
import plot 
import save
import routines
import RKKY_diagonalization_ky as H_diag
import numpy as np

from joblib import load
import argparse
import time as timer 

def main(parameters):

    name = 'eigen'
    for element in parameters:
        name += '_'+str(np.round(element,1))
    name += '.txt'
    coeffis = []
    #name = 'eigen data/eigen_40_40_0.5_1.6_0.3_0.2_1.0_15_20.txt'

    kvalues = list(np.arange(-np.pi, np.pi ,2*np.pi/(parameters[1]))) # N_y k-values
    positive_kvalues = [kvalues.index(element) for element in kvalues if round(element,6) > 0] #indices of positive k-values
    zero_kvalues = [kvalues.index(element) for element in kvalues if round(element,6) == 0]#indicies of zero k-values
    
    k = [positive_kvalues, zero_kvalues]

    eigen = load('eigen data/'+name)
    for k_value in range(parameters[1]):
        coeffis += [H_diag.operator_coefficients(eigen, k_value)]

    density = []

    for site in range(parameters[0]//2-2, parameters[0]//2+3):
        density += [o.density_of_states(eigen, coeffis, [site, parameters[0]], k)]
    #plot.density([eigen]*len(range(parameters[0]//2-2, parameters[0]//2+3)), density, site, parameters, labels=range(parameters[0]//2-2, parameters[0]//2+3))
    plot.density([eigen]*len(range(parameters[0]//2-2, parameters[0]//2+3)), density, site, parameters, range(parameters[0]//2-2, parameters[0]//2+3), 'site')
    
    return True

if __name__ == '__main__':
    start = timer.time()

    parser = argparse.ArgumentParser(description='Local density of states and gap equation for 2d superconductor (non-centro, incl. RKKY')

    parser.add_argument('sites_x', metavar='N_x', type=int, help='number of sites in x direcion') 
    parser.add_argument('sites_y', metavar='N_y', type=int, help='number of sites in y direcion') 
    parser.add_argument('chemical', metavar='mu', type=float, help='chemical potential') 
    parser.add_argument('attract', metavar='delta', type=float, help='electron pair interaction strength - singlet') 
    parser.add_argument('tri', metavar='delta_tri', type=float, help='electron pair interaction strength - triplet') 
    parser.add_argument('soc', metavar='gamma', type=float, help='spin-orbit-coupling strength') 
    parser.add_argument('rkky', metavar='jott', type=float, help='RKKY interaction strength')
    parser.add_argument('impure1_x', metavar='impurity1_x', type=int, help='position of first impurity x-coordinate')
    parser.add_argument('impure2_x', metavar='impurity2_x', type=int, help='position of second impurity x-coordinate')

    args = parser.parse_args()

    if main([args.sites_x, args.sites_y, args.chemical, args.attract, args.tri, args.soc, args.rkky, args.impure1_x, args.impure2_x]): 
        print('total duration ', round(timer.time()-start, 4))
