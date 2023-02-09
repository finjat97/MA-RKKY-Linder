## calculate the analytically derived spin structure with experimental values
import numpy as np
import time as timer
from joblib import dump, load
from tqdm import tqdm 
import matplotlib.pyplot as plt
from numba import jit, complex128, float64
from functools import reduce

t = 1 #hopping
cps_1 = 0.1 #singlet pairing potential
cps_3 = 0.1 #triplet pairing potential
gamma = 0.3 #SOC strength
jott = 3 #RKKY interaction strength
mu = 0.5 #chemical potential
sites =  11 #lattice sites

def fermi_dis(energy): # input and return: np.float64
    constant = 0.01 # k_B*T
    
    return 1/ (np.exp(energy/constant)+1)

# energy including spin orbit coupling: WARNING: I changed the calculation of |\gamma|
def xi(k, heli): # input: list of len 2, int; return: float
    return -2*t*(np.cos(k[0])+np.cos(k[1])) - mu + heli*abs((gamma**2)*(k[1]**2 + k[0]**2))

## this needs to be double checked because we are in helicity, not in spin basis
#supercondcuting gap (only for Rashba SOC and d parallel to SOC)
def delta(k): #input: list with length 2; return: 2d np.array
    return np.array([[(-k[1]+1j*(-k[0]))*cps_3,cps_1], [-cps_1, (k[1]+1j*(-k[0]))*cps_3]]) 

def E(k, heli):
    if heli > 0: index = 0
    else: index = 1

    return np.sqrt((xi(k, heli))**2 + abs(delta(k)[index][index])**2)

#components of eigenvectors
def norm(k,heli):
    if heli > 0: index = 0
    else: index = heli
    result = np.sqrt((E(k,heli) + (xi(k, heli)))**2 + abs(delta(k)[index][index])**2)
    if result == 0: return 0.0001
    else: return result

def nu(k, heli):
    normalisation = norm(k, heli)
    res = (E(k, heli) + (xi(k, heli)))/normalisation
    return res

def eta(k, heli):
    if heli > 0: index = 0
    else: index = 1
    normalisation = norm(k, heli)
    result = (delta(k)[index][index])/normalisation
    return result

#coefficients from the unitary transformation matrix S used in the Schrieffer-Wolff transformation

def A(k, k_2 , heli, heli_2):
    energy = (E(k, heli)-E(k_2, heli_2))
    if energy == 0: energy = 0.001
    return nu(k, heli).conjugate()*nu(k_2, heli_2)/energy

def B(k, k_2 , heli, heli_2):
    energy = (E([-x for x in k], heli)-E([-x for x in k_2], heli_2))
    if energy == 0: energy = 0.001
    return eta(k, heli)*eta(k_2, heli_2).conjugate()/energy

def C(k, k_2 , heli, heli_2):
    return eta(k, heli)*nu(k_2, heli_2)/(E([-x for x in k], heli)+E(k_2, heli_2))

def D(k, k_2 , heli, heli_2):
    return nu(k, heli).conjugate()*eta(k_2, heli_2).conjugate()/(E(k, heli)+E([-x for x in k_2], heli_2))

#coefficients from the RKKY hamiltonian
def a(k, k_2 , heli, heli_2):
    return nu(k, heli).conjugate()*nu(k_2, heli_2)

def b(k, k_2 , heli, heli_2):
    return eta(k, heli)*eta(k_2, heli_2).conjugate()

def c(k, k_2 , heli, heli_2):
    return eta(k, heli)*nu(k_2, heli_2)

def d(k, k_2 , heli, heli_2):
    return nu(k, heli).conjugate()*eta(k_2, heli_2).conjugate()

# energy prefactors for the spin structure
def F_minus(k, heli, heli_2):
    factor1, factor2 = [], []
    fermi = (fermi_dis(E(k[:2], heli))-fermi_dis(E(k[2:], heli_2)))
    for (f1,f2) in [[A,b], [B,a], [C,d], [D,c]]:
        factor1 += [f1(k[:2], k[2:], heli, heli_2)]
        factor2 += [f2(k[2:], k[:2], heli_2, heli)]
    
    return sum(map(lambda x,y: x*y*fermi, factor1, factor2))

def F_plus(k, heli, heli_2):
    result = 0
    fermi = (fermi_dis(E(k[:2], heli))+fermi_dis(E(k[2:], heli_2)))
    for (f1,f2) in [[A,b], [B,a], [C,d], [D,c]]:
        result +=fermi*f1(k[:2], k[2:], heli, heli_2)*f2(k[2:], k[:2], heli_2, heli)
    return result

def position(k, pos, pos_2):
    return np.exp(-1j*((k[:2][0]-k[2:][0])*(pos[0]-pos_2[0])+(k[:2][1]-k[2:][1])*(pos[1]-pos_2[1])))

def phase1(k):
    abs_k1 = np.sqrt(k[:2][0]**2+k[:2][1]**2)
    abs_k2 = np.sqrt(k[2:][0]**2+k[2:][1]**2)
    if abs_k1 == 0: abs_k1 = 0.001
    if abs_k2 == 0: abs_k2 = 0.001
    return ((k[2:][1]-1j*k[2:][0])/abs_k2)**2 + ((k[:2][1]+1j*k[:2][0])/abs_k1)**2

def phase2(k):
    abs_k1 = np.sqrt(k[:2][0]**2+k[:2][1]**2)
    abs_k2 = np.sqrt(k[2:][0]**2+k[2:][1]**2)
    if abs_k1 == 0: abs_k1 = 0.001
    if abs_k2 == 0: abs_k2 = 0.001
    return ((k[2:][1]-1j*k[2:][0])/abs_k2) * ((k[:2][1]+1j*k[:2][0])/abs_k1)

def phase3(k):
    abs_k1 = np.sqrt(k[:2][0]**2+k[:2][1]**2)
    abs_k2 = np.sqrt(k[2:][0]**2+k[2:][1]**2)
    if abs_k1 == 0: abs_k1 = 0.001
    if abs_k2 == 0: abs_k2 = 0.001
    return ((k[:2][1]-1j*k[:2][0])/abs_k1) - ((k[2:][1]+1j*k[2:][0])/abs_k2)


def J_negpos(F_pp, F_mm, F_pm, F_mp, F_p_pm, p1, p2, position):

    constants = ((jott/sites)**2)/4
    
    energy_x = -( (F_pp+F_mm)*(p1 + 2*p2) + (F_p_pm + F_mp)*(p1 - 2*p2) )
    energy_y = -( (F_pp + F_mm)*(p1-2*p2) - (F_p_pm - F_pm)*(p1 + 2*p2) )
    energy_z = 4* (F_mp + F_pm)

    return [constants*position*energy_x, constants*position*energy_y, constants*position*energy_z]

def J_pospos(F_pp, F_mm, F_pm, F_mp, position):
  
    constants = ((jott/sites)**2)/2

    energy_x = F_pp + F_mm + F_pm + F_mp
    energy_y = F_pp + F_mm - F_pm - F_mp
    energy_z = F_mp - 2* F_pm

    return [constants*position*energy_x, constants*position*energy_y, constants*position*energy_z]

def gamma_xy(F_pp, F_mm, F_mp, F_pm, p, position):

    constants = ((jott/sites)**2)/4
    energy = F_pp + F_mm + F_mp + F_pm

    return constants*position*energy*p

def D_y_pospos(F_pm, F_mp, p, position):

    constants = ((jott/sites)**2)/2
    energy = F_pm - F_mp 

    return -constants*position*energy*p

def D_y_negpos(F_pm, F_mp, p, position):

    constants = ((jott/sites)**2)/2

    phase = p.conjugate()

    energy = F_pm - F_mp

    return -constants*position*phase*energy

def main(compare= True, distance=2, plotting=True):
    all_F = []
    distances, D_y, Gamma, J_vec = [], [], [], []

    # constructing all combinations of k_values for later summation, type: 2d list of length 81, each entry of length 4; type(k_values)=np.ndarray
    k_values = np.array(np.meshgrid(np.arange(-np.pi, np.pi ,2*np.pi/(sites)),np.arange(-np.pi, np.pi ,2*np.pi/(sites)),np.arange(-np.pi, np.pi ,2*np.pi/(sites)),np.arange(-np.pi, np.pi ,2*np.pi/(sites)))).T.reshape(-1,4)
    
    if compare:
        spin_orientations = [[[0,1/2,0],[0,1/2,0]],[[0,1/2,0],[0,-1/2,0]]]
        # spin_orientations = [[[1/2,0,0],[1/2,0,0]],[[1/2,0,0],[0,0,1/2]],[[1/2,0,0],[0,0,-1/2]]]
        spin_position = [[1,0], [distance,0]]

    else:
        spin_position = [[1,0], [distance,0]]
        spin_orientations = np.array([[[0,1/2,0],[0,1/2,0]], [[1/2,0,0],[1/2,0,0]], [[0,-1/2,0],[0,1/2,0]], [[-1/2,0,0],[1/2,0,0]], [[0,1/2,0],[0,-1/2,0]], [[1/2,0,0],[-1/2,0,0]], 
        [[1/2,0,0],[0,1/2,0]], [[0,1/2,0],[1/2,0,0]], [[-1/2,0,0],[0,1/2,0]], [[0,-1/2,0],[1/2,0,0]], [[1/2,0,0],[0,-1/2,0]], [[0,1/2,0],[-1/2,0,0]],
        [[1/2,0,0],[0,0,1/2]], [[1/2,0,0],[0,0,-1/2]], [[-1/2,0,0],[0,0,1/2]], [[-1/2,0,0],[0,0,-1/2]], [[0,1/2,0],[0,0,1/2]], [[0,-1/2,0],[0,0,1/2]],
        [[0,1/2,0],[0,0,-1/2]], [[0,-1/2,0],[0,0,-1/2]], [[0,0,1/2],[0,0,1/2]],  [[0,0,-1/2],[0,0,1/2]],  [[0,0,1/2],[0,0,-1/2]],  [[0,0,-1/2],[0,0,-1/2]]])
    
    map_start = timer.time()

    # position dependent prefactor for all k_values; right now for 2d; WARNING: might be wrong calculation, because periodicity is not yet seen
    position_prefactor = list(map(position, k_values, [spin_position[0]]*len(k_values), [spin_position[1]]*len(k_values)))
    # energy prefactors for all different combinations of helicity that occur in the spin structure, for all k_values
    energy_pp = list(map(F_minus, k_values, [+1]*len(k_values), [+1]*len(k_values)))
    energy_mm = list(map(F_minus, k_values, [-1]*len(k_values), [-1]*len(k_values)))
    energy_pm = list(map(F_minus, k_values, [+1]*len(k_values), [-1]*len(k_values)))
    energy_mp = list(map(F_minus, k_values, [-1]*len(k_values), [+1]*len(k_values)))
    energy_p_pm = (map(F_plus, k_values, [+1]*len(k_values), [-1]*len(k_values)))
    # phases for all k_values
    phase1_factor = list(map(phase1, k_values))
    phase2_factor = (map(phase2, k_values))
    phase3_factor = list(map(phase3, k_values))

    J_pospos_result = list(map(J_pospos, energy_pp, energy_mm, energy_pm, energy_mp, position_prefactor))
    J_negpos_result = list(map(J_negpos, energy_pp, energy_mm, energy_pm, energy_mp, energy_p_pm, phase1_factor, phase2_factor, position_prefactor))
    gamma_result = list(map(gamma_xy, energy_pp, energy_mm, energy_pm, energy_mp, phase1_factor, position_prefactor))
    D_y_pospos_result = list(map(D_y_pospos, energy_pm, energy_mp, phase3_factor, position_prefactor))
    D_y_negpos_result = list(map(D_y_negpos, energy_pm, energy_mp, phase3_factor, position_prefactor))

    print('done with coefficients ', round(timer.time()- map_start,4))

    for spin in tqdm(spin_orientations):
        heisenberg = [spin[0][0]*spin[1][0] , spin[0][1]*spin[1][1] , spin[0][2]*spin[1][2]]
        free_energy = 0

        sumy = sum(map(lambda x: x*heisenberg[1], [item[1] for item in J_pospos_result]))
        sumx = sum(map(lambda x: x*heisenberg[0], [item[0] for item in J_pospos_result])) 
        sumz = sum(map(lambda x: x*heisenberg[2], [item[2] for item in J_pospos_result]))
        J_p_res = sumx + sumy + sumz
        
        sumy = sum(map(lambda x: x*heisenberg[1], [item[1] for item in J_negpos_result]))
        sumx = sum(map(lambda x: x*heisenberg[0], [item[0] for item in J_negpos_result])) 
        sumz = sum(map(lambda x: x*heisenberg[2], [item[2] for item in J_negpos_result]))
        J_n_res = sumx + sumy + sumz

        J_vec.append(J_p_res+J_n_res)

        D_p_res = sum(map(lambda x: x*(spin[0][2]*spin[1][0]- spin[0][0]*spin[1][2]), D_y_pospos_result))
        D_n_res = sum(map(lambda x: x*(spin[0][2]*spin[1][0]- spin[0][0]*spin[1][2]), D_y_negpos_result))
        
        D_y.append(D_n_res + D_p_res)

        g_res = sum(map(lambda x: x*(spin[0][1]*spin[1][0]+ spin[0][0]*spin[1][1]), gamma_result))
        Gamma.append(g_res)

        free_energy += J_n_res + J_p_res + D_n_res + D_p_res + g_res
        all_F.append(free_energy)
    
    configurations_label = []

    for version in range(len(spin_orientations)):

        version_label = []
        for site in range(len(spin_orientations[version])):
            index = [i for i, element in enumerate(spin_orientations[version][site]) if element != 0][0]
            if index == 0: 
                if spin_orientations[version][site][index] > 0: 
                    version_label += ['→']
                else: version_label += ['←']
            if index == 1:
                if spin_orientations[version][site][index] > 0: version_label += ['x']
                else: version_label += ['.']
            if index == 2:
                if spin_orientations[version][site][index] > 0: version_label += ['↑']
                else: version_label += ['↓']

        configurations_label.append(version_label)
    
    labels =[]
    for entry in configurations_label:
        labels += ['('+entry[0]+entry[1]+')']
    if plotting: 
        plt.plot(D_y, label=r'$D_y$')
        plt.plot(J_vec, label=r'$J$')
        plt.plot(Gamma, label=r'$\Gamma$')
        # plt.xticks(range(len(spin_orientations)), labels, rotation=45)
        plt.legend()
        plt.grid()
        # name = 'analytical/coefficient_relations'
        # for element in parameters:
        #     name += '_'+str(np.round(element,2))
        # name += '.png'
        # plt.savefig(name)
        plt.show()
        plt.clf()
        
        # plt.plot(all_F)
        # plt.xlabel('spin configuration')
        # plt.ylabel('free energy in 1/t')
        # plt.xticks(range(len(spin_orientations)), labels, rotation=45)
        # # plt.title('analytical free energy for different spin-orientations of two impurity spins \n for '+str(parameters[0])+r' sites with $U=$'+str(parameters[2])+r' and $V=$'+str(parameters[3])+', J='+str(parameters[5])+r', $\gamma=$'+str(round(parameters[4],2)))
        # plt.tight_layout()
        # # plt.legend()
        # plt.grid()
        # # name = 'analytical/analytical_spinstructure'
        # # for element in parameters:
        # #     name += '_'+str(np.round(element,2))
        # # name += '.png'
        # # plt.savefig(name)
        # plt.clf()

    return all_F, distances

if __name__ == '__main__':
    start = timer.time()

    result = main(compare=False)
    print('total duration ', round(timer.time()-start, 4))