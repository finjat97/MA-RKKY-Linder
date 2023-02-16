import numpy as np 
import matplotlib.pyplot as plt 
import time as timer
import argparse

from tqdm import tqdm 
from routines import spin_loop as sl 

def plot_gs(y_values, x_values, y_axis, para):
    labels =[]
    for entry in y_axis:
        labels += ['('+entry[0]+entry[1]+')']
    
    name = 'groundstate/gs'
    for element in para:
        name += '_'+str(np.round(element,2))
    name += '.png'

    plt.grid() 
    # for point in range(len(x_values)):
    #     plt.scatter(x_values[point], y_values[point])
    #     plt.scatter(x_values[point], y_values[point], marker=labels[y_values[point]])
    plt.scatter(x_values, y_values, label='groundstate')

    plt.yticks(range(len(y_axis)), labels, rotation=0)
    plt.xticks(x_values)
    plt.ylabel('spin configuration')
    plt.xlabel('impurity separation distance in a')
    #para = [sites_x, sites_y, mu, cps, tri, gamma, jott, 0, 0] 
    plt.title('Groundstate spin configuration for different distances \n for N = '+str(para[0]) + r', $\mu =$ '+str(para[2])+ r', $V=$'+str(para[3])+r', $U=$'+str(para[4]) + r', $\gamma=$'+str(para[5]) + r', $J=$'+str(para[6]) )
    plt.legend()
    plt.tight_layout()
    plt.savefig(name)
    #plt.show()

    return True

def main(sites_x, sites_y, mu, cps, tri, gamma, jott):

    #find minimum of free energy dependent on spin configuration for different separation distances of the impurities, parameters of the system stay the same
    spin_positions = []
    distances = []

    #determine different impurity positions depending on system size; starting at site 10 to avoid edge effects
    for y in range(10,sites_x-10, 1):
        spin_positions.append([10,y])
    
    parameters = [sites_x, sites_y, mu, cps, tri, gamma, jott, 0, 0] 
    kvalues = list(np.arange(-np.pi, np.pi ,2*np.pi/(sites_y))) 
    groundstates = []
    
    #go through different impurity separation distances
    for pos in tqdm(spin_positions):
        #calculating free energies for all spin configurations
        config, energy = sl(parameters, kvalues, pos)
        #finding the index of lowest free energy aka. groundstate; might be ambigious
        gs = (np.where(energy == np.min(energy)))[0]
        #collecting all groundstate indices in one list; all indices
        groundstates.append(gs)
        #collecting the distances with the correct degeneracy 
        distances.append([pos[1]-10]*len(gs))

    # flattening the lists to enable easy plotting
    groundstates = np.concatenate(groundstates).ravel()
    distances = np.concatenate(distances).ravel()

    # plotting the groundstate over the distance; plot is saved in 'groundstate'
    plot_gs(groundstates, distances, config, parameters[:-2])

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
    
    args = parser.parse_args()

    if main(args.sites_x, args.sites_y, args.chemical, args.attract, args.tri, args.rkky, args.soc): 
        print('total duration ', round(timer.time()-start, 4))
