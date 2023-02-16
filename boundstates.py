import numpy as np 
import observables as o 
import matplotlib.pyplot as plt 
import time as timer
import argparse
import RKKY_diagonalization_ky as H_diag

def plot_comp(x, y, sites, parameters):
    y1, y2, y3 = y
    p1, p2, p3 = sites
    x = np.arange(x[0][0][0],x[0][0][-1]+0.01,0.01)

    name = 'YSR states/bs'
    for element in parameters:
        name += '_'+str(np.round(element,2))
    name += '.png'

    plt.plot(x, y1, label= str(p1))
    plt.plot(x, y2, label= str(p2))
    plt.plot(x, y3, label= str(p3))
    plt.legend()
    plt.grid()
    plt.title('LDOS for neighbouring sites \n in '+str(round(parameters[0],2))+'x'+str(round(parameters[1],2))+', '+r' $U=$'+str(round(parameters[3],2))+r', $V=$'+str(round(parameters[4],2))+r', $\gamma=$'+str(round(parameters[5],2))+r', $J=$'+str(round(parameters[6],2))+r', $i=$'+str(round(parameters[7],2))+', '+str(round(parameters[8],2)))

    plt.xlabel('energy in 1/t')
    plt.ylabel('LDOS')
    plt.tight_layout()
    plt.savefig(name)
    plt.show()

    return True

def main(sites_x, sites_y, mu, cps, tri, jott, gamma, imp1, imp2):
    kvalues = list(np.arange(-np.pi, np.pi ,2*np.pi/(sites_y))) # N_y k-values \in [-pi, pi)
    positive_kvalues = [kvalues.index(element) for element in kvalues if round(element,6) > 0] #indices of positive k-values
    zero_kvalues = [kvalues.index(element) for element in kvalues if round(element,6) == 0]#indicies of zero k-values

    parameters = [sites_x, sites_y, mu, cps, tri, gamma, jott, imp1, imp2]
    spin_orientation = [[0,1/2,0], [0,-1/2, 0]]

    eigen = H_diag.diagonalize_hamiltonian(parameters, spin_orientation)
    # coeffis = H_diag.operator_coefficients(eigen, kvalues)
    def H_coeffi(x):
        return H_diag.operator_coefficients(eigen,x)
        
    coeffis = list(map(H_coeffi, list(range(parameters[1]))))
    
    ldos_imp = o.density_of_states(eigen, coeffis, [imp2, parameters[0]], [positive_kvalues, zero_kvalues])
    ldos_left = o.density_of_states(eigen, coeffis, [imp2-1, parameters[0]], [positive_kvalues, zero_kvalues])
    ldos_right = o.density_of_states(eigen, coeffis, [imp2+1, parameters[0]], [positive_kvalues, zero_kvalues])

    plot_comp(eigen, [ldos_imp, ldos_left, ldos_right], [imp2, imp2-1, imp2+1], parameters)    

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

    if main(args.sites_x, args.sites_y, args.chemical, args.attract, args.tri, args.rkky, args.soc, args.impure1_x, args.impure2_x): 
        print('total duration ', round(timer.time()-start, 4))
