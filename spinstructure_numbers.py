## calculate the analytically derived spin structure with experimental values
import numpy as np
import time as timer
from joblib import dump, load
from tqdm import tqdm 
import matplotlib.pyplot as plt
from numba import jit, complex128, float64
from functools import reduce

# sites = 11 #lattice sites
# t = 1 #hopping
# gamma = 0.1 #SOC strength
# jott = 2 #RKKY interaction strength
# mu = 1 #chemical potential
# cps_1 = 0.05 #singlet pairing strength, << t
# cps_3 = 0.0 #triplet pairing strength, <<t

spin = 1/2

def spin_orientation(winkel):
    res = np.array([np.round(spin*np.sin(winkel[:,0])*np.cos(winkel[:,1]),5), np.round(spin*np.sin(winkel[:,0])*np.sin(winkel[:,1]),5), np.round(spin*np.cos(winkel[:,0]),5)])
    res = res + 0
    return res

def main(sites, t, gamma, jott, mu, cps_1, cps_3, distance, compare=True, plotting=False):
    def fermi_dis(energy): # input and return: 1d array
        constant = 0.01 # k_B*T
        
        return 1/ (np.exp(energy/constant)+1)

    # energy including spin orbit coupling: WARNING: I changed the calculation of |\gamma|
    def xi(k, heli): # input: (2,k-space) array, int; return: (k space, ) array, float
        res = -2*t*(np.cos(k[:,0])+np.cos(k[:,1])) - mu + heli*np.sqrt((gamma**2)*(k[:,1]**2 + k[:,0]**2))
        return res

    ## this needs to be double checked because we are in helicity, not in spin basis
    #supercondcuting gap (only for Rashba SOC and d parallel to SOC)
    def delta(k): #input: list with length 2; return: 2d np.array
        #return np.array([[(-k[:,1]+1j*(-k[:,0]))*cps_3,cps_1], [-cps_1, (k[:,1]+1j*(-k[:,0]))*cps_3]]) 
        return [cps_1/2 + cps_3/2, cps_1/2-cps_3/2]

    def E(k, heli):
        if heli > 0: index = 0
        else: index = 1
        res =  np.sqrt((xi(k, heli))**2 + abs(delta(k)[index])**2)
        return res

    #components of eigenvectors
    def norm(k,heli):
        if heli > 0: index = 0
        else: index = heli
        result = np.sqrt((E(k,heli) + (xi(k, heli)))**2 + abs(delta(k)[index])**2)
        # print('delta= ', delta(k))
        # print('E= ', E(k,heli), (xi(k, heli)))
        # print('N= ', result)
        if result.all() == 0: result[result == 0] = 0.0001
        return result

    def nu(k, heli):
        normalisation = norm(k, heli)
        res = (E(k, heli) + (xi(k, heli)))/normalisation
        if res.all() == 0: res[res == 0] = 1
        res = res
        return res

    def eta(k, heli):
        if heli > 0: index = 0
        else: index = 1
        normalisation = norm(k, heli)
        result = (delta(k)[index])/normalisation
        # print('eta= ', result)
        return result

    #coefficients from the unitary transformation matrix S used in the Schrieffer-Wolff transformation

    def A(k, k_2 , heli, heli_2):
        energy = np.array(E(k, heli)-E(k_2, heli_2))
        if energy.all() == 0.0: energy[energy==0.] = 0.001
        return nu(k, heli).conjugate()*nu(k_2, heli_2)/energy

    def B(k, k_2 , heli, heli_2):
        energy = (E(-k, heli)-E(-k_2, heli_2))
        if energy.all() == 0.0: energy[energy==0.] = 0.001
        return eta(k, heli)*eta(k_2, heli_2).conjugate()/energy

    def C(k, k_2 , heli, heli_2):
        return eta(k, heli)*nu(k_2, heli_2)/(E(-k, heli)+E(k_2, heli_2))

    def D(k, k_2 , heli, heli_2):
        return nu(k, heli).conjugate()*eta(k_2, heli_2).conjugate()/(E(k, heli)+E(-k_2, heli_2))

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
        fermi = (fermi_dis(E(k[:,:2], heli))-fermi_dis(E(k[:,2:], heli_2)))
        for (f1,f2) in [[A,b], [B,a], [C,d], [D,c]]:
            factor1 += [f1(k[:,:2], k[:,2:], heli, heli_2)]
            factor2 += [f2(k[:,2:], k[:,:2], heli_2, heli)]
        res1 = np.array(factor1)*np.array(factor2)*fermi
        res = (sum(res1))
        # res = sum(map(lambda x,y: x*y*fermi, factor1, factor2))
        
        return res

    def F_minus_pp(k, heli, heli_2):
        factor1, factor2 = [], []
        fermi = (fermi_dis(E(k[:,:2], heli))-fermi_dis(E(k[:,2:], heli_2)))
        for (f1,f2) in [[A,a], [B,b], [C,c], [D,d]]:
            factor1 += [f1(k[:,:2], k[:,2:], heli, heli_2)]
            factor2 += [f2(k[:,2:], k[:,:2], heli_2, heli)]
        res = (sum(np.array(factor1)*np.array(factor2)*fermi))
        # res = sum(map(lambda x,y: x*y*fermi, factor1, factor2))
        return res

    def F_plus(k, heli, heli_2):
        result = 0
        fermi = (fermi_dis(E(k[:,:2], heli))+fermi_dis(E(k[:,2:], heli_2)))
        for (f1,f2) in [[A,b], [B,a], [C,d], [D,c]]:
            result +=fermi*f1(k[:,:2], k[:,2:], heli, heli_2)*f2(k[:,2:], k[:,:2], heli_2, heli)
        #result = sum(result)
        return result

    def position(k, pos, pos_2):
        return np.exp(-1j*((k[:,:2][:,0]-k[:,2:][:,0])*(pos[0]-pos_2[0])+(k[:,:2][:,1]-k[:,2:][:,1])*(pos[1]-pos_2[1])))

    def phase1(k):
        abs_k1 = np.sqrt(k[:,:2][:,0]**2+k[:,:2][:,1]**2)
        abs_k2 = np.sqrt(k[:,2:][:,0]**2+k[:,2:][:,1]**2)
        if abs_k1.all() == 0: abs_k1[abs_k1 == 0] = 0.001
        if abs_k2.all() == 0: abs_k2[abs_k2 == 0] = 0.001
        return ((k[:,2:][:,1]-1j*k[:,2:][:,0])/abs_k2)**2 + ((k[:,:2][:,1]+1j*k[:,:2][:,0])/abs_k1)**2

    def phase2(k):
        abs_k1 = np.sqrt(k[:,:2][:,0]**2+k[:,:2][:,1]**2)
        abs_k2 = np.sqrt(k[:,2:][:,0]**2+k[:,2:][:,1]**2)
        if abs_k1.all() == 0: abs_k1[abs_k1 == 0] = 0.001
        if abs_k2.all() == 0: abs_k2[abs_k2 == 0] = 0.001
        return ((k[:,2:][:,1]-1j*k[:,2:][:,0])/abs_k2) * ((k[:,:2][:,1]+1j*k[:,:2][:,0])/abs_k1)

    def phase3(k):
        abs_k1 = np.sqrt(k[:,:2][:,0]**2+k[:,:2][:,1]**2)
        abs_k2 = np.sqrt(k[:,2:][:,0]**2+k[:,2:][:,1]**2)
        if abs_k1.all() == 0: abs_k1[abs_k1 == 0] = 0.001
        if abs_k2.all() == 0: abs_k2[abs_k2 == 0] = 0.001
        return (((k[:,:2][:,1]-1j*k[:,:2][:,0])/abs_k1) - ((k[:,2:][:,1]+1j*k[:,2:][:,0])/abs_k2))


    def J_negpos(F_pp, F_mm, F_pm, F_mp, F_p_pm, p1, p2, position):

        constants = ((jott/sites)**2)/4
        
        energy_x = -( (F_pp+F_mm)*(p1 + 2*p2) + (F_p_pm + F_mp)*(p1 - 2*p2) )
        energy_y = -( (F_pp + F_mm)*(p1-2*p2) - (F_p_pm - F_pm)*(p1 + 2*p2) )
        energy_z = 4* (F_mp + F_pm)

        return np.array([constants*position*energy_x, constants*position*energy_y, constants*position*energy_z])

    def J_pospos(F_pp, F_mm, F_pm, F_mp, position):
    
        constants = ((jott/sites)**2)/2

        energy_x = F_pp + F_mm + F_pm + F_mp
        energy_y = F_pp + F_mm - F_pm - F_mp
        energy_z = F_mp - 2* F_pm

        return np.array([constants*position*energy_x, constants*position*energy_y, constants*position*energy_z])

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
    all_F = []
    D_y, Gamma, J_vec = [], [], []

    # constructing all combinations of k_values for later summation, type: 2d list of length 81, each entry of length 4; type(k_values)=np.ndarray
    k_values = np.array(np.meshgrid(np.arange(-np.pi, np.pi ,2*np.pi/(sites)),np.arange(-np.pi, np.pi ,2*np.pi/(sites)),np.arange(-np.pi, np.pi ,2*np.pi/(sites)),np.arange(-np.pi, np.pi ,2*np.pi/(sites)))).T.reshape(-1,4)

    if compare:
        two_spin = np.array([[[0,1/2,0],[0,1/2,0]],[[0,1/2,0],[0,-1/2,0]]])
        # spin_orientations = [[[1/2,0,0],[1/2,0,0]],[[1/2,0,0],[0,0,1/2]],[[1/2,0,0],[0,0,-1/2]]]
        spin_position = np.array([[0,0], [distance+1,0]])

    else:
        spin_position = np.array([[1,0], [distance+1,0]])
        step = 2 # number of intervals to dicretisize the spherical coordinates
        # find all angle combinations for discretisized spherical coordinates
        angle = np.array(np.meshgrid(np.arange(0, np.pi, np.pi/step), np.arange(-np.pi, np.pi, np.pi/step))).T.reshape(-1,2)
        # calculate all possible orientations based on those angle for one spin
        one_spin = spin_orientation(angle).T
        # find all possible combinations of two arrays with length of the angle combinations (aka. number of possible directions of one spin)
        combo = np.array(np.meshgrid(range(angle.shape[0]),range(angle.shape[0]))).T.reshape(-1,2)
        # find all possible ways to combine two spins with all directions allowed by the previous calculated angle combinations
        two_spin = np.unique(np.array([ one_spin[combo[:,0]] , one_spin[combo[:,1]] ]), axis=1) #shape = (2, possible directions **2, 3)

        spin_orientations = np.array([[[0,1/2,0],[0,1/2,0]], [[1/2,0,0],[1/2,0,0]], [[0,-1/2,0],[0,1/2,0]], [[-1/2,0,0],[1/2,0,0]], [[0,1/2,0],[0,-1/2,0]], [[1/2,0,0],[-1/2,0,0]], 
        [[1/2,0,0],[0,1/2,0]], [[0,1/2,0],[1/2,0,0]], [[-1/2,0,0],[0,1/2,0]], [[0,-1/2,0],[1/2,0,0]], [[1/2,0,0],[0,-1/2,0]], [[0,1/2,0],[-1/2,0,0]],
        [[1/2,0,0],[0,0,1/2]], [[1/2,0,0],[0,0,-1/2]], [[-1/2,0,0],[0,0,1/2]], [[-1/2,0,0],[0,0,-1/2]], [[0,1/2,0],[0,0,1/2]], [[0,-1/2,0],[0,0,1/2]],
        [[0,1/2,0],[0,0,-1/2]], [[0,-1/2,0],[0,0,-1/2]], [[0,0,1/2],[0,0,1/2]],  [[0,0,-1/2],[0,0,1/2]],  [[0,0,1/2],[0,0,-1/2]],  [[0,0,-1/2],[0,0,-1/2]]])
    
    # position dependent prefactor for all k_values; right now for 2d; WARNING: might be wrong calculation, because periodicity is not yet seen
    position_prefactor = position(k_values, spin_position[0], spin_position[1])
    # energy prefactors for all different combinations of helicity that occur in the spin structure, for all k_values
    energy_pp = F_minus(k_values, +1,+1) #np.array(list(map(F_minus, k_values, [+1]*len(k_values), [+1]*len(k_values))))
    energy_mm = F_minus(k_values, -1, -1)
    energy_pp_pp = F_minus_pp(k_values, +1, +1)
    energy_mm_pp = F_minus_pp(k_values, -1, -1)
    energy_pm = F_minus(k_values, +1, -1)
    energy_mp = F_minus(k_values, -1, +1)
    energy_pm_pp = F_minus_pp(k_values, +1, -1)
    energy_mp_pp = F_minus_pp(k_values, -1, +1)
    energy_p_pm = F_plus(k_values, +1, -1)
    # phases for all k_values
    phase1_factor = phase1(k_values)
    phase2_factor = phase2(k_values)
    phase3_factor = phase3(k_values)

    J_pospos_result = J_pospos(energy_pp_pp, energy_mm_pp, energy_pm_pp, energy_mp_pp, position_prefactor)
    J_negpos_result = J_negpos(energy_pp, energy_mm, energy_pm, energy_mp, energy_p_pm, phase1_factor, phase2_factor, position_prefactor)
    gamma_result = gamma_xy(energy_pp, energy_mm, energy_pm, energy_mp, phase1_factor, position_prefactor)
    D_y_pospos_result = D_y_pospos(energy_pm_pp, energy_mp_pp, phase3_factor, position_prefactor)
    D_y_negpos_result = D_y_negpos(energy_pm, energy_mp, phase3_factor, position_prefactor)
    

    heisenberg = two_spin[0] * two_spin[1] #shape = (3, possible directions **2)
    dm = two_spin[0][:,2] * two_spin[1][:,0] - two_spin[0][:,0] * two_spin[1][:,2] # shape= (possible directions **2)
    rest = two_spin[0][:,1] * two_spin[1][:,0] + two_spin[0][:,0]*two_spin[1][:,1] # shape= (possible directions **2)

    J_vec = sum(J_pospos_result.T) + sum(J_negpos_result.T)
    D_y = sum(D_y_negpos_result) + sum(D_y_pospos_result)
    Gamma = sum(gamma_result)

    all_F = sum((J_vec * heisenberg).T) + D_y * dm  + Gamma *rest 
    #print(min(all_F), combo[np.where(all_F == min(all_F))], angle)

    if plotting: 
        
        labels = [str(row[:3])+ ' '+str(row[3:]) for row in np.concatenate(two_spin, axis=1)]
    
        plt.plot(D_y*dm, label=r'$D_y$')
        plt.plot(sum((J_vec * heisenberg).T), label=r'$J$')
        plt.plot(Gamma*rest, label=r'$\Gamma$')
        plt.xticks(range(len(two_spin[0])), labels, rotation=90)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        parameters = [sites, mu, cps_1, cps_3, gamma, jott, spin_position[0][0], spin_position[1][0]]
        name = 'JDG_analytical/coefficient_relations'
        for element in parameters:
            name += '_'+str(np.round(element,2))
        name += '.png'
        plt.savefig(name)
        plt.show()
        plt.clf()
        
        plt.plot(all_F)
        plt.xlabel('spin configuration')
        plt.ylabel('free energy in 1/t')
        plt.xticks(range(len(two_spin[0])), labels, rotation=90)
        plt.title('analytical free energy for different spin-orientations of two impurity spins \n for '+str(parameters[0])+r' sites with $U=$'+str(parameters[2])+r' and $V=$'+str(parameters[3])+', J='+str(parameters[5])+r', $\gamma=$'+str(round(parameters[4],2)))
        plt.tight_layout()
        plt.grid()
        name = 'analytical spinstructure/analytical_spinstructure'
        for element in parameters:
            name += '_'+str(np.round(element,2))
        name += '.png'
        plt.savefig(name)
        plt.clf()
        
        # plt.plot(all_F)
        # plt.xlabel('spin configuration')
        # plt.ylabel('free energy in 1/t')
        # plt.xticks(range(len(spin_orientations)), labels, rotation=45)
        # plt.title('analytical free energy for different spin-orientations of two impurity spins \n for '+str(parameters[0])+r' sites with $U=$'+str(parameters[2])+r' and $V=$'+str(parameters[3])+', J='+str(parameters[5])+r', $\gamma=$'+str(round(parameters[4],2)))
        # plt.tight_layout()
        # plt.grid()
        # name = 'analytical spinstructure/analytical_spinstructure'
        # for element in parameters:
        #     name += '_'+str(np.round(element,2))
        # name += '.png'
        # plt.savefig(name)
        # plt.clf()

    return all_F, J_vec, D_y, Gamma, two_spin #spin_orientations

if __name__ == '__main__':
    start = timer.time()

    result = main(21, 1, 0.2, 2, 1, 0.01, 0.05, 2, compare=False, plotting=True)
    print('total duration ', round(timer.time()-start, 4))