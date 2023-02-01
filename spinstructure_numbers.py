## calculate the analytically derived spin structure with experimental values
import numpy as np
import time as timer
from joblib import dump, load
from tqdm import tqdm 
import matplotlib.pyplot as plt
import numba as nb

t = 1 #hopping
cps_1 = 0 #singlet pairing potential
cps_3 = 0 #triplet pairing potential
gamma = 0.3 #SOC strength
jott = 3 #RKKY interaction strength
mu = 0.5 #chemical potential
sites =  20 #lattice sites

#fermi distribution
def fermi_dis(energy):
    constant = 0.01 # k_B*T

    return 1/ (np.exp(energy/constant)+1)

#energy including spin orbit coupling
# def xi(k_x, k_y, heli):
#     return -2*t*(np.cos(k[0])+np.cos(k[1])) - mu + heli*abs(gamma*(k[1] - k[0]))

#supercondcuting gap   
# def delta(k[0],k[1]):
#     return [[0,cps_1],[-cps_1, 0],[(-k[1]+1j*(-k[0]))*cps_3,(0)*cps_3],[(0)*cps_3, (k[1]+1j*(-k[0]))*cps_3]]

#energy including spin orbit coupling and superconductivity
def E(k, heli):
    if heli > 0: index = 0
    else: index = heli
    return np.sqrt((-2*t*(np.cos(k[0])+np.cos(k[1])) - mu + heli*abs(gamma*(k[1] - k[0])))**2 + abs([[0,cps_1],[-cps_1, 0],[(-k[1]+1j*(-k[0]))*cps_3,(0)*cps_3],[(0)*cps_3, (k[1]+1j*(-k[0]))*cps_3]][index][index])**2)

#components of eigenvectors
def norm(k,heli):
    if heli > 0: index = 0
    else: index = heli
    return np.sqrt((E(k,heli) + (-2*t*(np.cos(k[0])+np.cos(k[1])) - mu + heli*abs(gamma*(k[1] - k[0]))))**2 + abs([[0,cps_1],[-cps_1, 0],[(-k[1]+1j*(-k[0]))*cps_3,(0)*cps_3],[(0)*cps_3, (k[1]+1j*(-k[0]))*cps_3]][index][index])**2)

def nu(k, heli):
    normalisation = norm(k, heli)
    if normalisation == 0: normalisation = 0.001
    return (E(k, heli) + (-2*t*(np.cos(k[0])+np.cos(k[1])) - mu + heli*abs(gamma*(k[1] - k[0]))))/normalisation

def eta(k, heli):
    if heli > 0: index = 0
    else: index = heli
    normalisation = norm(k, heli)
    if normalisation == 0: normalisation = 0.001
    return ([[0,cps_1],[-cps_1, 0],[(-k[1]+1j*(-k[0]))*cps_3,(0)*cps_3],[(0)*cps_3, (k[1]+1j*(-k[0]))*cps_3]][index][index])/normalisation

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

#energy prefactors for the spin structure, factor_x is the coefficient that the energy belongs to
def F_minus(factor_1, factor_2, k, k_2, heli, heli_2):
    return (fermi_dis(E(k, heli))-fermi_dis(E(k_2, heli_2)))*factor_1(k, k_2, heli, heli_2)*factor_2(k_2, k, heli_2, heli)

def F_plus(factor_1, factor_2, k, k_2, heli, heli_2):
    return (fermi_dis(E(k, heli))+fermi_dis(E(k_2, heli_2)))*factor_1(k, k_2, heli, heli_2)*factor_2(k_2, k, heli_2, heli)

#spin structure components
def D_y_pospos(k_x, k_y ,k_2_x, k_2_y, pos, pos_2):

    k = [k_x,k_y]
    k_2 = [k_2_x, k_2_y]

    constants = ((jott/sites)**2)/2
    position = np.exp(-1j*((k[0]-k_2[0])*(pos[0]-pos_2[0])+(k[1]-k_2[1])*(pos[1]-pos_2[1]))) #position in 2D

    abs_k = np.sqrt(k[0]**2+k[1]**2)
    abs_k_2 = np.sqrt(k_2[0]**2+k_2[1]**2)
    if abs_k == 0: abs_k = 0.001
    if abs_k_2 == 0: abs_k_2 = 0.001

    phase = ((k[1]-1j*k[0])/abs_k) - ((k_2[1]+1j*k_2[0])/abs_k_2)

    energy = 0
    for element in [[A,a], [B,b], [C,d], [D,c]]:
        f1 = element[0]
        f2 = element[1]
        energy += F_minus(f1, f2, k, k_2, +1, -1) - F_minus(f1, f2, k, k_2, -1, +1)

    return -constants*position*energy*phase
# @nb.jit(nopython=True)
def J_pospos(k_x, k_y ,k_2_x, k_2_y, pos, pos_2):

    k = [k_x,k_y]
    k_2 = [k_2_x, k_2_y]
    constants = ((jott/sites)**2)/2
    position = np.exp(-1j*((k[0]-k_2[0])*(pos[0]-pos_2[0])+(k[1]-k_2[1])*(pos[1]-pos_2[1]))) #position in 2D

    energy_x, energy_y, energy_z = 0, 0, 0
    for element in [[A,a], [B,b], [C,d], [D,c]]:
        f1 = element[0]
        f2 = element[1]
        energy_x += F_minus(f1, f2, k,k_2, +1, +1)+ F_minus(f1, f2, k,k_2, -1, -1)+ F_minus(f1, f2, k,k_2, +1, -1)+F_minus(f1, f2, k,k_2, -1, +1)
        energy_y += F_minus(f1, f2, k,k_2, +1, +1)+F_minus(f1, f2, k,k_2, -1, -1)-F_minus(f1, f2, k,k_2, +1, -1)-F_minus(f1, f2, k,k_2, +1, -1)
        energy_z += -2* F_minus(f1, f2, k,k_2, -1, +1)- 2*F_minus(f1, f2, k,k_2, +1, -1)

    return [constants*position*energy_x, constants*position*energy_y, constants*position*energy_z]

def D_y_negpos(k_x, k_y ,k_2_x, k_2_y, pos, pos_2):

    k = [k_x,k_y]
    k_2 = [k_2_x, k_2_y]
    constants = ((jott/sites)**2)/2
    position = np.exp(-1j*((k[0]-k_2[0])*(pos[0]-pos_2[0])+(k[1]-k_2[1])*(pos[1]-pos_2[1]))) #position in 2D

    abs_k = np.sqrt(k[0]**2+k[1]**2)
    abs_k_2 = np.sqrt(k_2[0]**2+k_2[1]**2)
    if abs_k == 0: abs_k = 0.001
    if abs_k_2 == 0: abs_k_2 = 0.001

    phase = ((k[1]+1j*k[0])/abs_k) - ((k_2[1]-1j*k_2[0])/abs_k_2)

    energy = 0
    for element in [[A,b], [B,a], [C,d], [D,c]]:
        f1 = element[0]
        f2 = element[1]
        energy += F_minus(f1, f2, k, k_2, +1, -1) - F_minus(f1, f2, k, k_2, -1, +1)

    return -constants*position*phase*energy

# @nb.jit(nopython=True)
def J_negpos(k_x, k_y ,k_2_x, k_2_y, pos, pos_2):

    k = [k_x,k_y]
    k_2 = [k_2_x, k_2_y]
    constants = ((jott/sites)**2)/4
    position = np.exp(-1j*((k[0]-k_2[0])*(pos[0]-pos_2[0])+(k[1]-k_2[1])*(pos[1]-pos_2[1]))) #position in 2D

    abs_k = np.sqrt(k[0]**2+k[1]**2)
    abs_k_2 = np.sqrt(k_2[0]**2+k_2[1]**2)
    if abs_k == 0: abs_k = 0.001
    if abs_k_2 == 0: abs_k_2 = 0.001

    phase_1 = ((k_2[1]-1j*k_2[0])/abs_k_2)**2 + ((k[1]+1j*k[0])/abs_k)**2
    phase_2 = ((k_2[1]-1j*k_2[0])/abs_k_2) * ((k[1]+1j*k[0])/abs_k)

    energy_x, energy_y, energy_z = 0, 0, 0
    
    for element in [[A,b], [B,a], [C,d], [D,c]]:
        f1 = element[0]
        f2 = element[1]
        energy_x += -((F_minus(f1,f2,k,k_2, +1, +1)+ F_minus(f1,f2,k,k_2, -1, -1))*(phase_1 + 2*phase_2) + (F_plus(f1, f2, k,k_2, +1, -1)+F_minus(f1,f2,k,k_2, -1, +1))*(phase_1 - 2* phase_2))
        energy_y += -((F_minus(f1,f2,k,k_2, +1, +1)+F_minus(f1,f2,k,k_2, -1, -1))*(phase_1 - 2*phase_2)-(F_plus(f1, f2, k,k_2, +1, -1)-F_minus(f1,f2,k,k_2, +1, -1))*(phase_1 + 2*phase_2))
        energy_z += 4* F_minus(f1,f2,k,k_2, -1, +1)+4*F_minus(f1,f2,k,k_2, +1, -1)

    return [constants*position*energy_x, constants*position*energy_y, constants*position*energy_z]

def gamma_xy(k_x, k_y ,k_2_x, k_2_y, pos, pos_2):

    k = [k_x,k_y]
    k_2 = [k_2_x, k_2_y]
    constants = ((jott/sites)**2)/4
    position = np.exp(-1j*((k[0]-k_2[0])*(pos[0]-pos_2[0])+(k[1]-k_2[1])*(pos[1]-pos_2[1]))) #position in 2D

    abs_k = np.sqrt(k[0]**2+k[1]**2)
    abs_k_2 = np.sqrt(k_2[0]**2+k_2[1]**2)
    if abs_k == 0: abs_k = 0.001
    if abs_k_2 == 0: abs_k_2 = 0.001

    phase = -((k_2[1]-1j*k_2[0])/abs_k_2)**2 + ((k[1]+1j*k[0])/abs_k)**2

    energy = 0
    for element in [[A,b], [B,a], [C,d], [D,c]]:
        f1 = element[0]
        f2 = element[1]
        energy += F_minus(f1, f2, k, k_2, +1, +1)+ F_minus(f1, f2, k,k_2, -1, -1)+ F_minus(f1, f2, k,k_2, +1, -1)+F_minus(f1, f2, k,k_2, -1, +1)

    return constants*position*energy*phase


def main(compare=False, distance=0, plotting=False):

    spin_position = [[sites//2,0],[0,sites//4]] #position of impurity spins
    parameters = [sites, mu, cps_1, cps_3, gamma, jott] + spin_position

    # print(load(name), 'load')

    all_F, D_y, J_vec, Gamma, k_1_x, k_1_y, k_2_x, k_2_y = [], [], [], [] ,[], [], [], []
    
    start_time = timer.time()

    for item_x in list(np.arange(-np.pi, np.pi ,2*np.pi/(sites))):
        for item_y in list(np.arange(-np.pi, np.pi ,2*np.pi/(sites))):
            k_1_x += [item_x]
            k_1_y += [item_y]
    for item_x in list(np.arange(-np.pi, np.pi ,2*np.pi/(sites))):
        for item_y in list(np.arange(-np.pi, np.pi ,2*np.pi/(sites))):
            k_2_x += [item_x]
            k_2_y += [item_y]
    #k = list(np.arange(-np.pi, np.pi ,2*np.pi/(sites)))
    k_time = timer.time()
    print('k loops: ', round(k_time-start_time,4))

    if compare:
        spin_orientations = [[[0,1/2,0],[0,1/2,0]],[[0,1/2,0],[0,-1/2,0]]]
        spin_position = [[1,0], [distance,0]]

    else:
        spin_orientations = [[[0,1/2,0],[0,1/2,0]], [[1/2,0,0],[1/2,0,0]], [[0,-1/2,0],[0,1/2,0]], [[-1/2,0,0],[1/2,0,0]], [[0,1/2,0],[0,-1/2,0]], [[1/2,0,0],[-1/2,0,0]], 
        [[1/2,0,0],[0,1/2,0]], [[0,1/2,0],[1/2,0,0]], [[-1/2,0,0],[0,1/2,0]], [[0,-1/2,0],[1/2,0,0]], [[1/2,0,0],[0,-1/2,0]], [[0,1/2,0],[-1/2,0,0]],
        [[1/2,0,0],[0,0,1/2]], [[1/2,0,0],[0,0,-1/2]], [[-1/2,0,0],[0,0,1/2]], [[-1/2,0,0],[0,0,-1/2]], [[0,1/2,0],[0,0,1/2]], [[0,-1/2,0],[0,0,1/2]],
        [[0,1/2,0],[0,0,-1/2]], [[0,-1/2,0],[0,0,-1/2]], [[0,0,1/2],[0,0,1/2]],  [[0,0,-1/2],[0,0,1/2]],  [[0,0,1/2],[0,0,-1/2]],  [[0,0,-1/2],[0,0,-1/2]]]

    for version in tqdm(range(len(spin_orientations))):
        spin = spin_orientations[version]
        heisenberg = [spin[0][0]*spin[1][0] , spin[0][1]*spin[1][1] , spin[0][2]*spin[1][2]]
        # for cps in np.arange(0,3,0.5):
        #     global J 
        #     J = cps
                        # free_energy = D_y_pospos(k_1, k_2, spin_position[0], spin_position[1])*(spin[0][2]*spin[1][0]- spin[0][0]*spin[1][2])
        D_y_pospos_result = list(map(D_y_pospos, k_1_x,k_1_y,k_2_x,k_2_y, [spin_position[0]]*len(k_1_x), [spin_position[1]]*len(k_1_x)))
        D_y_pospos_result = sum(list(map(lambda x: x*(spin[0][2]*spin[1][0]- spin[0][0]*spin[1][2]), D_y_pospos_result))) 
        free_energy = D_y_pospos_result   
        # D_pp_time = timer.time()
        # # print('D_pp calc: ', round(D_pp_time-k_time,4))
                        # free_energy += sum(np.multiply(J_pospos(k_1,k_2, spin_position[0], spin_position[1]), heisenberg))
        J_pospos_result = list(map(J_pospos, k_1_x,k_1_y,k_2_x,k_2_y, [spin_position[0]]*len(k_1_x), [spin_position[1]]*len(k_1_x)))
        J_pospos_result = sum(list(map(lambda x,y,z: x+y+z, list(map(lambda x: x*heisenberg[0], [item[0] for item in J_pospos_result])),list(map(lambda x: x*heisenberg[1], [item[1] for item in J_pospos_result])),list(map(lambda x: x*heisenberg[2], [item[2] for item in J_pospos_result])))))
        free_energy += J_pospos_result    
        # # # J_pp_time = timer.time()
        # # print('J_pp calc: ', round(J_pp_time-D_pp_time,4))            
                        # free_energy += D_y_negpos(k_1, k_2, spin_position[0], spin_position[1])*(spin[0][2]*spin[1][0]- spin[0][0]*spin[1][2])
        D_y_negpos_result = list(map(D_y_negpos, k_1_x,k_1_y,k_2_x,k_2_y, [spin_position[0]]*len(k_1_x), [spin_position[1]]*len(k_1_x)))
        D_y_negpos_result = sum(list(map(lambda x: x*(spin[0][2]*spin[1][0]- spin[0][0]*spin[1][2]), D_y_negpos_result))) 
        free_energy += D_y_negpos_result
        # # # D_np_time = timer.time()
        # # print('D_np calc: ', round(D_np_time-J_pp_time,4))
                        # free_energy += sum(np.multiply(J_negpos(k_1,k_2, spin_position[0], spin_position[1]), heisenberg))  
        J_negpos_result = list(map(J_negpos, k_1_x,k_1_y,k_2_x,k_2_y, [spin_position[0]]*len(k_1_x), [spin_position[1]]*len(k_1_x)))
        J_negpos_result = sum(list(map(lambda x,y,z: x+y+z, list(map(lambda x: x*heisenberg[0], [item[0] for item in J_negpos_result])),list(map(lambda x: x*heisenberg[1], [item[1] for item in J_negpos_result])),list(map(lambda x: x*heisenberg[2], [item[2] for item in J_negpos_result])))))      
        free_energy += J_negpos_result     
        # # # J_np_time = timer.time()
        # # print('J_np calc: ', round(J_np_time-D_np_time,4))           
                        # free_energy += gamma_xy(k_1, k_2, spin_position[0], spin_position[1])*(spin[0][1]*spin[1][0]+ spin[0][0]*spin[1][1])
        gamma_xy_result = list(map(gamma_xy, k_1_x,k_1_y,k_2_x,k_2_y, [spin_position[0]]*len(k_1_x), [spin_position[1]]*len(k_1_x)))
        gamma_xy_result = sum(list(map(lambda x: x*(spin[0][1]*spin[1][0]+ spin[0][0]*spin[1][1]), gamma_xy_result))) 
        free_energy += gamma_xy_result
        # # # gamma_time = timer.time()
        # # print('gamma calc: ', round(gamma_time-J_np_time,4))

        # # # one_time = timer.time()
        # # print('one version: ', round(one_time-k_time,4))
        
        D_y += [D_y_negpos_result+D_y_pospos_result]
        J_vec += [J_negpos_result+J_pospos_result]
        Gamma += [gamma_xy_result]
        all_F += [free_energy]
        if free_energy.imag > 10**(-30): print(free_energy, spin)
    
    version_time = timer.time()
    print('spin loop: ', round(version_time-k_time, 4))

    name = 'free energy analytical/eigen'
    for element in parameters:
        name += '_'+str(np.round(element,1))
    name += '.txt'
    dump(all_F, name, compress=2)
    # print('gamma', Gamma)
    # print('d_y', D_y)
    # print('J', J_vec)
    
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
        name = 'coefficient_relations'
        for element in parameters:
            name += '_'+str(np.round(element,2))
        name += '.png'
        plt.savefig(name)
        plt.show()
        plt.clf()
        
        plt.plot(all_F)
        plt.xlabel('spin configuration')
        plt.ylabel('free energy in 1/t')
        plt.xticks(range(len(spin_orientations)), labels, rotation=45)
        plt.title('analytical free energy for different spin-orientations of two impurity spins \n for '+str(parameters[0])+r' sites with $U=$'+str(parameters[2])+r' and $V=$'+str(parameters[3])+', J='+str(parameters[5])+r', $\gamma=$'+str(round(parameters[4],2)))
        plt.tight_layout()
        # plt.legend()
        plt.grid()
        name = 'analytical_spinstructure'
        for element in parameters:
            name += '_'+str(np.round(element,2))
        name += '.png'
        plt.savefig(name)
        plt.clf()

    # print('max D_y: ', max(D_y))

    return all_F, parameters

if __name__ == '__main__':
    start = timer.time()

    result = main(compare=True)
    print('total duration ', round(timer.time()-start, 4))