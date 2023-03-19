import numpy as np 
import time as timer
#from joblib import dump, load
from tqdm import tqdm 
import matplotlib.pyplot as plt
import scipy.io as sio


spin = 1/2

def spin_orientation(winkel):
    res = np.array([np.round(spin*np.sin(winkel[:,0])*np.cos(winkel[:,1]),5), np.round(spin*np.sin(winkel[:,0])*np.sin(winkel[:,1]),5), np.round(spin*np.cos(winkel[:,0]),5)])
    res = res + 0
    return res

def main(sites, t, gamma, jott, mu, cps_1, cps_3, distance, compare=True, plotting=False):
    all_F = np.zeros(sites)

    def fermi(energy): # input and return: 1d array
        constant = 0.01 # k_B*T
        
        return 1/ (np.exp(energy/constant)+1)
    
    def xi(k, heli): # input: (2,k-space) array, int; return: (k space, ) array, float
        res = -2*t*(np.cos(k[:,0])+np.cos(k[:,1])) - mu + heli*np.sqrt((gamma**2)*(k[:,1]**2 + k[:,0]**2))
        return res

    ## this needs to be double checked because we are in helicity, not in spin basis
    #supercondcuting gap (only for Rashba SOC and d parallel to SOC)
    def delta(k): #input: list with length 2; return: 2d np.array
        #return np.array([[(-k[:,1]+1j*(-k[:,0]))*cps_3,cps_1], [-cps_1, (k[:,1]+1j*(-k[:,0]))*cps_3]]) 
        return [cps_1/2 + cps_3/2, cps_1/2-cps_3/2]

    
    def energy(k, heli):
        if heli > 0: index = 0
        else: index = 1
        res =  np.sqrt((xi(k, heli))**2 + abs(delta(k)[index])**2)
        return res
    
     
    def a(k, k_2 , heli, heli_2):
        return nu(k, heli).conjugate()*nu(k_2, heli_2)

    def b(k, k_2 , heli, heli_2):
        return eta(k, heli)*eta(k_2, heli_2).conjugate()

    def c(k, k_2 , heli, heli_2):
        return eta(k, heli)*nu(k_2, heli_2)

    def d(k, k_2 , heli, heli_2):
        return nu(k, heli).conjugate()*eta(k_2, heli_2).conjugate()

    def norm(k,heli):
        if heli > 0: index = 0
        else: index = heli
        result = np.sqrt((energy(k,heli) + (xi(k, heli)))**2 + abs(delta(k)[index])**2)
        if result.all() == 0: result[result == 0] = 0.0001
        return result

    def nu(k, heli):
        normalisation = norm(k, heli)
        res = (energy(k, heli) + (xi(k, heli)))/normalisation
        if res.all() == 0: res[res == 0] = 1
        res = res
        return res

    def eta(k, heli):
        if heli > 0: index = 0
        else: index = 1
        normalisation = norm(k, heli)
        result = (delta(k)[index])/normalisation
        return result


    def energy1(a1, a2, b1, b2, c1, c2, d1, d2, e1, e2):
        if 0 in e2-e1: e2[np.where(e2-e1==0)] += 10**(-5)
        energy = a1*a2 * (fermi(e1)- fermi(e2))/(e2-e1) + b1*b2* (fermi(e2)- fermi(e1))/(e1-e2)+ c1*d2*(-1)*(fermi(e2)+ fermi(e1))/(e1+e2) + d1*c2*(fermi(e1)-fermi(e2))/(e1+e2)
        return energy

    def energy2(a1, a2, b1, b2, c1, c2, d1, d2, e1, e2):
        if 0 in e2-e1: 
            e2[np.where(e2-e1==0)] += 10**(-5)
        energy = a1*b2* (fermi(e1)- fermi(e2))/(e2-e1) + b1*a2* (fermi(e2)- fermi(e1))/(e1-e2) + c1*d2*(fermi(e2)+ fermi(e1))/(e1+e2) + d1*c2*(-1)*(fermi(e1)-fermi(e2))/(e1+e2)
        return energy

    def I(mplus, mminus, phi1, phi1prime):
        ising1 = np.array([mplus + mminus, -(mplus + mminus), 1j*(mminus-mplus)])
        ising2 = np.array([2*phi1/phi1, phi1prime**2 + np.conj(phi1)**2, - phi1prime**2 + np.conj(phi1)**2])
        return ising1, ising2
    
    def D(k):
        x = np.array([ 2j * k[:,1]/np.sqrt(k[:,0]**2 + k[:,1]**2), - 2j * k[:,3]/np.sqrt(k[:,2]**2 + k[:,3]**2) ])
        y = np.array( [ 2j * k[:,0]/np.sqrt(k[:,0]**2 + k[:,1]**2), - 2j * k[:,2]/np.sqrt(k[:,0]**2 + k[:,1]**2)] )        
        z = np.zeros((2,k.shape[0]))
        return np.array([x,y,z])
    
    def Gamma(phi1, phi1prime, phi2, mplus, mminus):

        xy = np.array([1j*(phi1prime**2 - np.conj(phi1)**2), 1j*(mminus - mplus)])
        yx = xy
        yz = np.array([ 2j*np.conj(phi1), -2j*phi1prime ])
        zx = np.array([ phi2*phi1prime - np.conj(phi1) , phi2*np.conj(phi1)+phi1prime])
        zy = np.array([ -1j*(phi2*phi1prime + np.conj(phi1)), 1j*(phi2*np.conj(phi1)+phi1prime) ])

        return xy, yx, yz, zx, zy
    
    def position(k, pos, pos_2):
        return np.exp(-1j*((k[:,0]-k[:,2])*(pos[0]-pos_2[0])+(k[:,1]-k[:,3])*(pos[1]-pos_2[1])))

    
    k_values = np.array(np.meshgrid(np.arange(-np.pi, np.pi+2*np.pi/(sites) ,2*np.pi/(sites)),np.arange(-np.pi, np.pi+2*np.pi/(sites) ,2*np.pi/(sites)), np.arange(-np.pi, np.pi+2*np.pi/(sites) ,2*np.pi/(sites)),np.arange(-np.pi, np.pi+2*np.pi/(sites) ,2*np.pi/(sites)))).T.reshape(-1,4)

    energies = np.array( [ energy(k_values[:,:2], +1), energy(k_values[:,:2], -1) ] )
    energiesprime = np.array( [ energy(k_values[:,2:], +1), energy(k_values[:,2:], -1) ] )

    a1 = np.array([a(k_values[:,:2], k_values[:,2:], +1, +1), a(k_values[:,:2], k_values[:,2:], +1, -1), a(k_values[:,:2], k_values[:,2:], -1, +1), a(k_values[:,:2], k_values[:,2:], -1, -1)])
    a2 = np.array([a(k_values[:,2:], k_values[:,:2], +1, +1), a(k_values[:,2:], k_values[:,:2], +1, -1), a(k_values[:,2:], k_values[:,:2], -1, +1), a(k_values[:,2:], k_values[:,:2], -1, -1)])
    a3 = np.array([a(-k_values[:,:2], -k_values[:,2:], +1, +1), a(-k_values[:,:2], -k_values[:,2:], +1, -1), a(-k_values[:,:2], -k_values[:,2:], -1, +1), a(-k_values[:,:2], -k_values[:,2:],-1, -1)])

    b1 = np.array([b(k_values[:,:2], k_values[:,2:], +1, +1), b(k_values[:,:2], k_values[:,2:], +1, -1), b(k_values[:,:2], k_values[:,2:], -1, +1), b(k_values[:,:2], k_values[:,2:], -1, -1)])
    b2 = np.array([b(k_values[:,2:], k_values[:,:2], +1, +1), b(k_values[:,2:], k_values[:,:2], +1, -1), b(k_values[:,2:], k_values[:,:2], -1, +1), b(k_values[:,2:], k_values[:,:2], -1, -1)])
    b3 = np.array([b(-k_values[:,:2], -k_values[:,2:], +1, +1), b(-k_values[:,:2], -k_values[:,2:], +1, -1), b(-k_values[:,:2], -k_values[:,2:], -1, +1), b(-k_values[:,:2], -k_values[:,2:],-1, -1)])

    c1 = np.array([c(k_values[:,:2], k_values[:,2:], +1, +1), c(k_values[:,:2], k_values[:,2:], +1, -1), c(k_values[:,:2], k_values[:,2:], -1, +1), c(k_values[:,:2], k_values[:,2:], -1, -1)])
    c2 = np.array([c(k_values[:,2:], k_values[:,:2], +1, +1), c(k_values[:,2:], k_values[:,:2], +1, -1), c(k_values[:,2:], k_values[:,:2], -1, +1), c(k_values[:,2:], k_values[:,:2], -1, -1)])
    c3 = np.array([c(-k_values[:,:2], -k_values[:,2:], +1, +1), c(-k_values[:,:2], -k_values[:,2:], +1, -1), c(-k_values[:,:2], -k_values[:,2:], -1, +1), c(-k_values[:,:2], -k_values[:,2:],-1, -1)])

    d1 = np.array([d(k_values[:,:2], k_values[:,2:], +1, +1), d(k_values[:,:2], k_values[:,2:], +1, -1), d(k_values[:,:2], k_values[:,2:], -1, +1), d(k_values[:,:2], k_values[:,2:], -1, -1)])
    d2 = np.array([d(k_values[:,2:], k_values[:,:2], +1, +1), d(k_values[:,2:], k_values[:,:2], +1, -1), d(k_values[:,2:], k_values[:,:2], -1, +1), d(k_values[:,2:], k_values[:,:2], -1, -1)])
    d3 = np.array([d(-k_values[:,:2], -k_values[:,2:], +1, +1), d(-k_values[:,:2], -k_values[:,2:], +1, -1), d(-k_values[:,:2], -k_values[:,2:], -1, +1), d(-k_values[:,:2], -k_values[:,2:],-1, -1)])

    
    
    # E1_ppp = energy1(a1[0], a2[0], b1[0], b2[0], c1[0], c2[0], d1[0], d2[0], energies[0], energiesprime[0]) +  energy1(a1[3], a2[3], b1[3], b2[3], c1[3], c2[3], d1[3], d2[3], energies[1], energiesprime[1]) +  energy1(a1[1], a2[1], b1[1], b2[1], c1[1], c2[1], d1[1], d2[1], energies[0], energiesprime[1]) +  energy1(a1[2], a2[2], b1[2], b2[2], c1[2], c2[2], d1[2], d2[2], energies[1], energiesprime[0])
    # E1_pmm = energy1(a1[0], a2[0], b1[0], b2[0], c1[0], c2[0], d1[0], d2[0], energies[0], energiesprime[0]) +  energy1(a1[3], a2[3], b1[3], b2[3], c1[3], c2[3], d1[3], d2[3], energies[1], energiesprime[1]) -  energy1(a1[1], a2[1], b1[1], b2[1], c1[1], c2[1], d1[1], d2[1], energies[0], energiesprime[1]) -  energy1(a1[2], a2[2], b1[2], b2[2], c1[2], c2[2], d1[2], d2[2], energies[1], energiesprime[0])
    # E1_mpm = energy1(a1[0], a2[0], b1[0], b2[0], c1[0], c2[0], d1[0], d2[0], energies[0], energiesprime[0]) -  energy1(a1[3], a2[3], b1[3], b2[3], c1[3], c2[3], d1[3], d2[3], energies[1], energiesprime[1]) +  energy1(a1[1], a2[1], b1[1], b2[1], c1[1], c2[1], d1[1], d2[1], energies[0], energiesprime[1]) -  energy1(a1[2], a2[2], b1[2], b2[2], c1[2], c2[2], d1[2], d2[2], energies[1], energiesprime[0])
    # E1_mmp = energy1(a1[0], a2[0], b1[0], b2[0], c1[0], c2[0], d1[0], d2[0], energies[0], energiesprime[0]) -  energy1(a1[3], a2[3], b1[3], b2[3], c1[3], c2[3], d1[3], d2[3], energies[1], energiesprime[1]) -  energy1(a1[1], a2[1], b1[1], b2[1], c1[1], c2[1], d1[1], d2[1], energies[0], energiesprime[1]) +  energy1(a1[2], a2[2], b1[2], b2[2], c1[2], c2[2], d1[2], d2[2], energies[1], energiesprime[0])
    energies = np.array([energies[0], energies[0], energies[1], energies[1]])
    energiesprime = np.array([energiesprime[0], energiesprime[1], energiesprime[0], energiesprime[1]])

    E1_ppp = np.sum(energy1(a1, a2, b1, b2, c1, c2, d1, d2, energies, energiesprime)*np.array([1,1,1,1]).reshape(4,1), axis=0)
    E1_pmm = np.sum(energy1(a1, a2, b1, b2, c1, c2, d1, d2, energies, energiesprime)*np.array([1,-1,-1,1]).reshape(4,1), axis=0)
    E1_mpm = np.sum(energy1(a1, a2, b1, b2, c1, c2, d1, d2, energies, energiesprime)*np.array([1,1,-1,-1]).reshape(4,1), axis=0)
    E1_mmp = np.sum(energy1(a1, a2, b1, b2, c1, c2, d1, d2, energies, energiesprime)*np.array([1,-1,1,-1]).reshape(4,1), axis=0)

    # E2_ppp = energy2(a1[0], a3[0], b1[0], b3[0], c1[0], c3[0], d1[0], d3[0], energies[0], energiesprime[0]) +  energy2(a1[3], a3[3], b1[3], b3[3], c1[3], c3[3], d1[3], d3[3], energies[1], energiesprime[1]) +  energy2(a1[1], a3[1], b1[1], b3[1], c1[1], c3[1], d1[1], d3[1], energies[0], energiesprime[1]) +  energy2(a1[2], a3[2], b1[2], b3[2], c1[2], c3[2], d1[2], d3[2], energies[1], energiesprime[0])
    # E2_mpm = energy2(a1[0], a3[0], b1[0], b3[0], c1[0], c3[0], d1[0], d3[0], energies[0], energiesprime[0]) -  energy2(a1[3], a3[3], b1[3], b3[3], c1[3], c3[3], d1[3], d3[3], energies[1], energiesprime[1]) +  energy2(a1[1], a3[1], b1[1], b3[1], c1[1], c3[1], d1[1], d3[1], energies[0], energiesprime[1]) -  energy2(a1[2], a3[2], b1[2], b3[2], c1[2], c3[2], d1[2], d3[2], energies[1], energiesprime[0])
    # E2_mmp = energy2(a1[0], a3[0], b1[0], b3[0], c1[0], c3[0], d1[0], d3[0], energies[0], energiesprime[0]) -  energy2(a1[3], a3[3], b1[3], b3[3], c1[3], c3[3], d1[3], d3[3], energies[1], energiesprime[1]) -  energy2(a1[1], a3[1], b1[1], b3[1], c1[1], c3[1], d1[1], d3[1], energies[0], energiesprime[1]) +  energy2(a1[2], a3[2], b1[2], b3[2], c1[2], c3[2], d1[2], d3[2], energies[1], energiesprime[0])

    E2_ppp = np.sum(energy2(a1, a3, b1, b3, c1, c3, d1, d3, energies, energiesprime)*np.array([1,1,1,1]).reshape(4,1), axis=0)
    E2_mpm = np.sum(energy2(a1, a3, b1, b3, c1, c3, d1, d3, energies, energiesprime)*np.array([1,1,-1,-1]).reshape(4,1), axis=0)
    E2_mmp = np.sum(energy2(a1, a3, b1, b3, c1, c3, d1, d3, energies, energiesprime)*np.array([1,-1,1,-1]).reshape(4,1), axis=0)



    phi1 = (-k_values[:,1] + 1j* k_values[:,0])/np.sqrt(k_values[:,1]**2 + k_values[:,0]**2)
    phi1prime = (-k_values[:,3] + 1j* k_values[:,2])/np.sqrt(k_values[:,3]**2 + k_values[:,2]**2)
    phi2 = (-k_values[:,3] + 1j* k_values[:,2])/np.sqrt(k_values[:,3]**2 + k_values[:,2]**2) * (-k_values[:,1] - 1j* k_values[:,0])/np.sqrt(k_values[:,1]**2 + k_values[:,0]**2)
    mplus= phi1prime * phi1
    mminus = np.conj(mplus)

    if compare:
        two_spin = np.array([[[0,1/2,0],[0,1/2,0]],[[0,1/2,0],[0,-1/2,0]]])
        # spin_orientations = [[[1/2,0,0],[1/2,0,0]],[[1/2,0,0],[0,0,1/2]],[[1/2,0,0],[0,0,-1/2]]]
        spin_position = np.array([[0,0], [distance+1,0]])

    else:
        spin_position = np.array([[1,0], [distance+1,0]])
        step = 4 # number of intervals to dicretisize the spherical coordinates
        # find all angle combinations for discretisized spherical coordinates
        angle = np.array(np.meshgrid(np.arange(0, np.pi, np.pi/step), np.arange(-np.pi, np.pi, np.pi/step))).T.reshape(-1,2)
        # calculate all possible orientations based on those angle for one spin
        one_spin = spin_orientation(angle).T
        # find all possible combinations of two arrays with length of the angle combinations (aka. number of possible directions of one spin)
        combo = np.array(np.meshgrid(range(angle.shape[0]),range(angle.shape[0]))).T.reshape(-1,2)
        # find all possible ways to combine two spins with all directions allowed by the previous calculated angle combinations
        two_spin = np.unique(np.array([ one_spin[combo[:,0]] , one_spin[combo[:,1]] ]), axis=1) #shape = (2, possible directions **2, 3)
    

    position_prefactor = position(k_values, spin_position[0], spin_position[1])


    heisenberg = sum(2*E1_ppp*position_prefactor)
    ising = I(mplus, mminus, phi1, phi1prime)*position_prefactor
    ising = sum((ising[0]*E1_pmm + ising[1]*E2_ppp).T)
    dm = D(k_values)*position_prefactor
    dm = sum((dm[:,0]* E1_mpm + dm[:,1] * E1_mmp).T) 

    tensor = Gamma(phi1, phi1prime, phi2, mplus, mminus)*position_prefactor
    tensor = np.array([np.zeros(E2_ppp.shape[0]),tensor[0][0]*E2_ppp + tensor[0][1]*E1_pmm, np.zeros(E2_ppp.shape[0]),tensor[1][0]*E2_ppp + tensor[1][1]*E1_pmm, np.zeros(E2_ppp.shape[0]), tensor[2][0]*E2_mpm + tensor[2][1]*E2_mmp, tensor[3][0]*E2_mpm + tensor[3][1]*E2_mmp, tensor[4][0]*E2_mpm + tensor[4][1]*E2_mmp, np.zeros(E2_ppp.shape[0])])
    tensor = sum(tensor.T).reshape(3,3)

    # calculate the different spin factors, direct product, cross product and tensor product
    scalar = two_spin[0] * two_spin[1] #shape = (possible directions **2,3)
    cross = np.cross(two_spin[0], two_spin[1]) # shape= (possible directions **2, 3)
    #calculate tensor product of the two spins to get all 9 combinations of their components, for all possible configurations
    complete = np.array([np.zeros(two_spin.shape[1]),two_spin[0,:,0]* two_spin[1,:,1], np.zeros(two_spin.shape[1]), two_spin[0,:,1]* two_spin[1,:,0],np.zeros(two_spin.shape[1]), two_spin[0,:,1]* two_spin[1,:,2], two_spin[0,:,2]* two_spin[1,:,0], two_spin[0,:,2]* two_spin[1,:,1], np.zeros(two_spin.shape[1])])# shape=
    complete = complete.reshape(complete.shape[1],3,3) # shape=(possible directions ** 2, 3,3)

    # multiply coefficients with spin combinations for all possible configurations, shape = (possible config,)
    heisenberg = sum((heisenberg*scalar).T)
    ising = sum((ising * scalar).T)
    dm = sum((dm*cross).T)
    tensor = np.einsum('ij,...ij', tensor, complete)

    # final free energy is sum of all terms
    # DO I NEED THE MINUS SIGN?
    all_F = -(jott/sites)**2*(heisenberg + ising + dm + tensor)

    # plt.plot([min(all_F), max(all_F)], label=[two_spin[:,np.where(all_F == min(all_F))], two_spin[:,np.where(all_F == max(all_F))]])
    # plt.grid()
    # plt.legend()
    # plt.show()

    return all_F, two_spin

if __name__ == '__main__':
    start = timer.time()
    result = []
    for distance in range(2,6):
        # sites, t, gamma, jott, mu, cps_1, cps_3, distance
        output = main(7, 1, 0.7, 2, 1, 0, 0, distance, compare=False, plotting=True)
        result += [output[0]]
        spins = output[1]
    
    spins = spins.reshape(spins.shape[1], 2, 3)
    
    y = [min(element) for element in result]
    gs = [np.where(element == min(element))[0][0] for element in result]
    x = list(range(2,24))
    
    savedict = {'gs': gs, 'dis':x, 'spin':spins[gs]}
    sio.savemat('gs_data_num_25_1_0.7_2_1_0_0.mat', savedict)
   
    plt.plot(x,y)
    plt.plot(x,gs)
    plt.grid()
    plt.show()
    print('total duration ', round(timer.time()-start, 4))