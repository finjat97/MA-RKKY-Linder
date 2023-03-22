import numpy as np 
import matplotlib.pyplot as plt 
import time as timer
import argparse
import scipy.io as sio


from tqdm import tqdm 
from routines import spin_loop as sl 
from joblib import dump

def plot_gs(y_values, x_values, y_axis, para):

    labels = makelabel(y_axis)
    
    name = 'groundstate/gs'
    for element in para:
        name += '_'+str(np.round(element,2))
    name += '.png'

    def find_alphabeta(config):
        config = config*2
        
        sign3 = np.arccos(np.sum(config[:,0]* config[:,1], axis=1))
        sign3 = np.cos(sign3)/np.absolute(np.cos(sign3))

        config = np.absolute(config)
        gamma = np.arccos(np.sum(config[:,0]* config[:,1], axis=1))
        gamma = np.round((gamma - np.pi/4)/ (np.pi/4), 4)

        return gamma, sign3

    gamma, sign = find_alphabeta(y_axis[y_values])   

    positive = np.where(sign > 0)
    negative = np.where(sign < 0)

    labels = [str(row[0])+ ' '+str(row[1]) for row in np.round(y_axis[y_values],3)]

    plt.scatter(x_values[positive], gamma[positive], color='red', label='pos' )
    plt.scatter(x_values[negative], gamma[negative], color='blue', label='neg')

    #plt.yticks(range(len(labels)), labels, rotation=0)
    plt.xticks(x_values)
    plt.ylabel('colinearity of spin configuration')
    plt.xlabel('impurity separation distance in a')
    #para = [sites_x, sites_y, mu, cps, tri, gamma, jott, 0, 0] 
    plt.title('Groundstate spin configuration for different distances \n for N = '+str(para[0]) + r', $\mu =$ '+str(para[2])+ r', $U=$'+str(para[3])+r', $V=$'+str(para[4]) + r', $\gamma=$'+str(para[5]) + r', $J=$'+str(para[6]) )
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(name)
    plt.show()

    return True

def makelabel(config):

    labels = [str(row[:3])+ ' '+str(row[3:]) for row in np.concatenate(config, axis=1)]
    
    return labels

def main(sites_x, sites_y, mu, cps, tri, gamma, jott):

    #find minimum of free energy dependent on spin configuration for different separation distances of the impurities, parameters of the system stay the same
    spin_positions = []
    distances = []

    #determine different impurity positions depending on system size; starting at site 10 to avoid edge effects
    first_site = 15
    for y in range(first_site,sites_x-first_site, 1):
        spin_positions.append([first_site-2,y])
    
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
        distances.append([pos[1]-(first_site-2)]*len(gs))

    # flattening the lists to enable easy plotting
    groundstates = np.concatenate(groundstates).ravel()
    distances = np.concatenate(distances).ravel()

    # plotting the groundstate over the distance; plot is saved in 'groundstate'
    plot_gs(groundstates, distances, config, parameters[:-2])

    name = 'groundstate/gs'
    for element in parameters[:-2]:
        name += '_'+str(np.round(element,2))
    name += '.txt'
    name2 = 'groundstate/dis'
    for element in parameters[:-2]:
        name2 += '_'+str(np.round(element,2))
    name2 += '.txt'
    name3 = 'groundstate/spin'
    for element in parameters[:-2]:
        name3 += '_'+str(np.round(element,2))
    name3 += '.txt'
    name4 = 'groundstate/gs'
    for element in parameters[:-2]:
        name4 += '_'+str(np.round(element,2))
    name4 += '.mat'

    dump(groundstates, name, compress=2)
    dump(distances, name2, compress=2)
    dump(config, name3, compress=2)


    savedict = {'gs': groundstates, 'dis':distances, 'spin':config}
    sio.savemat(name4, savedict)

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

    if main(args.sites_x, args.sites_y, args.chemical, args.attract, args.tri, args.soc, args.rkky): 
        print('total duration ', round(timer.time()-start, 4))
