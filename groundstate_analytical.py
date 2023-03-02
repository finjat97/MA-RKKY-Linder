import numpy as np 
import matplotlib.pyplot as plt 
import time as timer
import argparse

from tqdm import tqdm 
from spinstructure_numbers import main as an

def plot_gs(y_values, x_values, labels, para):
    
    name = 'groundstate/gs_an'
    for element in para:
        name += '_'+str(np.round(element,2))
    name += '.png'

    plt.grid() 
    # for point in range(len(x_values)):
    #     plt.scatter(x_values[point], y_values[point])
    #     plt.scatter(x_values[point], y_values[point], marker=labels[y_values[point]])
    plt.scatter(x_values, y_values, label='groundstate')

    plt.yticks(range(len(labels)), labels, rotation=0)
    plt.xticks(x_values, rotation=45)
    plt.ylabel('spin configuration')
    plt.xlabel('impurity separation distance in a')
    #para = [sites_x, sites_y, mu, cps, tri, gamma, jott, 0, 0] 
    plt.title('Groundstate spin configuration for different distances \n for N = '+str(para[0]) + r', $\mu =$ '+str(para[1])+ r', $V=$'+str(para[2])+r', $U=$'+str(para[3]) + r', $\gamma=$'+str(para[4]) + r', $J=$'+str(para[5]) )
    plt.legend()
    plt.tight_layout()
    plt.savefig(name)
    plt.clf()
    #plt.show()

    return True

def plot_c(y1, y2, y3, x, labels, para):
    
    name = 'groundstate/c_an'
    for element in para:
        name += '_'+str(np.round(element,2))
    name += '.png'

    plt.grid() 
    # for point in range(len(x_values)):
    #     plt.scatter(x_values[point], y_values[point])
    #     plt.scatter(x_values[point], y_values[point], marker=labels[y_values[point]])
    plt.plot(x,y1, label=r'$J$ ')
    plt.plot(x,y2, label=r'$D_y$')
    plt.plot(x,y3, label=r'$\Gamma$')

    plt.xticks(x, rotation=45)
    plt.ylabel('energy')
    plt.xlabel('impurity separation distance in a')
    #para = [sites_x, sites_y, mu, cps, tri, gamma, jott, 0, 0] 
    plt.title('Energy of spin coefficients for different distances \n for N = '+str(para[0]) + r', $\mu =$ '+str(para[1])+ r', $V=$'+str(para[2])+r', $U=$'+str(para[3]) + r', $\gamma=$'+str(para[4]) + r', $J=$'+str(para[5]) )
    plt.legend()
    plt.tight_layout()
    plt.savefig(name)
    plt.clf()
    #plt.show()

    return True

def main(sites_x, mu, cps, tri, gamma, jott):

    #find minimum of free energy dependent on spin configuration for different separation distances of the impurities, parameters of the system stay the same
    distances = []
    parameters = [sites_x, mu, cps, tri, gamma, jott, 0, 0] 
    groundstates, Jc, Dc, Gc = [], [], [], []
    
    #go through different impurity separation distances
    for pos in tqdm(range(1,sites_x)):
        #calculating free energies for all spin configurations
        all_F, J_vec, D_y, Gamma, spin_orientations = an(sites_x, 1, gamma, jott, mu, cps, tri, pos, compare=False, plotting=False)
        #finding the index of lowest free energy aka. groundstate; might be ambigious
        gs = (np.where(all_F == np.min(all_F)))[0]
        #collecting all groundstate indices in one list; all indices
        groundstates.append(gs)
        #collecting values of coefficients 
        Jc.append(J_vec)
        Dc.append(D_y)
        Gc.append(Gamma)
        #collecting the distances with the correct degeneracy 
        distances.append([pos]*len(gs))
    # flattening the lists to enable easy plotting
    groundstates = np.concatenate(groundstates).ravel()
    distances = np.concatenate(distances).ravel()

    #make signs for the labeling of spin configurations
    # configurations_label = []
        
    # for version in (range(len(spin_orientations))):
    #     version_label = []

    #     for site in range(len(spin_orientations[version])):
    #         index = [i for i, element in enumerate(spin_orientations[version][site]) if element != 0][0]
    #         if index == 0: 
    #             if spin_orientations[version][site][index] > 0: 
    #                 version_label += ['→']
    #             else: version_label += ['←']
    #         if index == 1:
    #             if spin_orientations[version][site][index] > 0: version_label += ['x']
    #             else: version_label += ['.']
    #         if index == 2:
    #             if spin_orientations[version][site][index] > 0: version_label += ['↑']
    #             else: version_label += ['↓']

    #     configurations_label.append(version_label)

    # labels =[]
    # for entry in configurations_label:
    #     labels += ['('+entry[0]+entry[1]+')']
    labels = [str(row[:3])+ ' '+str(row[3:]) for row in np.concatenate(spin_orientations, axis=1)]

    # plotting the groundstate over the distance; plot is saved in 'groundstate_analytical'
    plot_gs(groundstates, distances, labels, parameters[:-2])

    #plotting the spin coefficients ove the distance; plot is saved in 'groundstate_anlytical'
    plot_c(np.array(Jc), np.array(Dc), np.array(Gc), range(1,sites_x), labels, parameters[:-2])

    return True

if __name__ == '__main__':
    start = timer.time()

    parser = argparse.ArgumentParser(description='Local density of states and gap equation for 2d superconductor (non-centro, incl. RKKY')

    parser.add_argument('sites_x', metavar='N_x', type=int, help='number of sites in x direcion') 
    parser.add_argument('chemical', metavar='mu', type=float, help='chemical potential') 
    parser.add_argument('attract', metavar='delta', type=float, help='electron pair interaction strength - singlet') 
    parser.add_argument('tri', metavar='delta_tri', type=float, help='electron pair interaction strength - triplet') 
    parser.add_argument('soc', metavar='gamma', type=float, help='spin-orbit-coupling strength') 
    parser.add_argument('rkky', metavar='jott', type=float, help='RKKY interaction strength')
    
    args = parser.parse_args()

    if main(args.sites_x, args.chemical, args.attract, args.tri, args.soc, args.rkky,): 
        print('total duration ', round(timer.time()-start, 4))
