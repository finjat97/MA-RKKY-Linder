import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm 
import argparse

if __name__ == '__main__':
    ## get arguments from terminal / program call
    parser = argparse.ArgumentParser(description='Coefficients of effective spin-structure for unconventional superconductor')

    parser.add_argument('sites_x', metavar='N_x', type=int, help='number of sites ') 
    parser.add_argument('chemical', metavar='mu', type=float, help='chemical potential') 
    parser.add_argument('attract', metavar='delta', type=float, help='electron pair interaction strength - singlet') 
    parser.add_argument('tri', metavar='delta_tri', type=float, help='electron pair interaction strength - triplet') 
    parser.add_argument('soc', metavar='gamma', type=float, help='spin-orbit-coupling strength') 
    
    args = parser.parse_args()

    ## set constants 
    N = args.sites_x
    mu = -args.chemical
    t = 1
    gamma = args.soc
    cps_1 = args.attract
    cps_3 = args.tri
    # N, mu, t, gamma, cps_1, cps_3 = 21, -0.5, 1, 0.1, 0.1, 0.01
    delta = [cps_1/2 + cps_3/2, cps_1/2-cps_3/2]

    beta = 10

    # contructing array with all possible combinations of (-pi/N,..., pi/N) for a 2d vector
    kvalues = np.array(np.meshgrid(np.arange(-np.pi,np.pi-np.pi/N,np.pi/N),np.arange(-np.pi,np.pi-np.pi/N,np.pi/N))).T.reshape(-1,2)
    
    # array containing all distances
    rvec = np.arange(0,15.1,0.1)

    ##############################################################
    ###### functions needed for calculation of coefficients ######
    ##############################################################

    def fermi_dis(energy):
        return 1/(1+np.exp(beta*energy))

    #energy of normal metal with SOC
    def xi(k, heli):
        return -2*t*(np.cos(k[:,0])+np.cos(k[:,1])) - mu + heli*np.sqrt((gamma**2)*(k[:,1]**2 + k[:,0]**2))

    #energy of superconductor with SOC
    def E(k, heli):
        if heli > 0: index = 0
        else: index = 1
        res = np.sqrt( (xi(k,heli))**2 + abs(delta[index])**2)
        return res
    
    #normalisation of k vector for specific helicity
    def norm(k,heli):
        if heli > 0: index = 0
        else: index = heli
        res = np.sqrt((E(k,heli) + (xi(k, heli)))**2 + abs(delta[index])**2)
        res[res == 0] = 0.0001
        return res
    
    ## components of eigenvectors aka. transformation coefficients
    def eta(k, heli):
        res = (E(k, heli) + (xi(k, heli)))/norm(k, heli)
        return res + 0
    
    def nu(k, heli):
        if heli > 0: index = 0
        else: index = 1
        return (delta[index])/norm(k, heli)
   
    ## the actual spin structure coefficients
    # Ising
    def I(mplus, mminus, phi1, phi1prime, phi2):
        ising1 = np.array([mplus + mminus, -2*phi2, -(phi1prime**2+ np.conj(phi1)**2)])
        ising2 = np.array([-(mplus+mminus), -2*phi2, (phi1prime**2+ np.conj(phi1)**2)])
        ising3 = np.array([-(phi2+np.conj(phi2)), -(phi2 + np.conj(phi2)) ,1+phi2**2])
        return np.array([ising1, ising2, ising3])
    
    # Dzyaloshinski-Moriya
    def DM(k1, k2, phi1, phi1prime, phi2):
        # k1n = np.sqrt(k1[:,0]**2 + k1[:,1]**2)
        # k2n = np.sqrt(k2[:,0]**2 + k2[:,1]**2)
        x = np.array([ np.zeros(phi2.shape[0]) + 1j*(phi1+np.conj(phi1)),  -1j*(phi1prime+np.conj(phi1prime)), 1j*(phi2*phi1prime+np.conj(phi1)), -1j*(phi2*np.conj(phi1)+phi1prime)])
        y = np.array([np.zeros(phi2.shape[0]) +np.conj(phi1)-phi1, phi1prime-np.conj(phi1prime), phi2*phi1prime - np.conj(phi1), phi2*np.conj(phi1)-phi1prime] )        
        z = np.zeros(y.shape)

        return np.array([x,y,z])
    
    # 3x3 tensor with missing interactions
    def Gamma(phi1, phi1prime, phi2, mplus, mminus):
        xy = np.array([1j*(mminus-mplus), 1j*(phi1prime**2 - np.conj(phi1)**2)])
        
        return xy
    
    ############################################################################
    ###### summation over k-space ######
    ############################################################################
    all_F = 0
    J_all, I_all, D_all, G_all = 0,0,0,0
    diff = np.zeros((1,rvec.shape[0]), dtype='complex128')

    for k1 in tqdm(kvalues):
        k1 = k1.reshape(1,2)

        ## construct all energy terms that er reused in the different spin structure coefficients
        energies = np.array( [ E(k1, +1), E(k1, -1) ] )
        energiesprime = np.array( [ E(kvalues, +1), E(kvalues, -1) ] )

        neg_energies = np.array( [ E(-k1, +1), E(-k1, -1) ] )
        neg_energiesprime = np.array( [ E(-kvalues, +1), E(-kvalues, -1) ] )

        energies = np.array([energies[0], energies[0], energies[1], energies[1]])
        energiesprime = np.array([energiesprime[0], energiesprime[1], energiesprime[0], energiesprime[1]])

        neg_energies = np.array([neg_energies[0], neg_energies[0], neg_energies[1], neg_energies[1]])
        neg_energiesprime = np.array([neg_energiesprime[0], neg_energiesprime[1], neg_energiesprime[0], neg_energiesprime[1]])

        em = energies - energiesprime
        nem = neg_energies - neg_energiesprime
        pnep = energies + neg_energiesprime
        npep = neg_energies + energiesprime

        ## construct all transformation coefficients stemming from the transformation betweeen helicity and eigenbasis of the superconductor

        eta_k = np.asarray([eta(k1, +1), eta(k1, +1), eta(k1, -1), eta(k1, -1)])
        eta_kp = np.asarray([eta(kvalues, +1), eta(kvalues, -1), eta(kvalues, +1), eta(kvalues, -1)])
        eta_mk = np.asarray([eta(-k1, +1), eta(-k1, +1), eta(-k1, -1), eta(-k1, -1)])
        eta_mkp = np.asarray([eta(-kvalues, +1), eta(-kvalues, -1), eta(-kvalues, +1), eta(-kvalues, -1)])

        nu_k = np.asarray([nu(k1, +1), nu(k1, +1), nu(k1, -1), nu(k1, -1)]) 
        nu_kp = np.asarray([nu(kvalues, +1), nu(kvalues, -1), nu(kvalues, +1), nu(kvalues, -1)])
        nu_mk = np.asarray([nu(-k1, +1), nu(-k1, +1), nu(-k1, -1), nu(-k1, -1)]) 
        nu_mkp = np.asarray([nu(-kvalues, +1), nu(-kvalues, -1), nu(-kvalues, +1), nu(-kvalues, -1)])

        ## construct all fermi distribution combinations
        fe = fermi_dis(energies)
        feprime = fermi_dis(energiesprime)
        
        neg_fe = fermi_dis(neg_energies)
        neg_feprime = fermi_dis(neg_energiesprime)

        fm = fe - feprime
        fm[em==0] = 0

        nnfm = neg_fe - neg_feprime
        nnfm[nem==0] = 0

        npfp = neg_fe + feprime
        pnfp = fe + neg_feprime

        #make sure to not divied by zero. the terms will end up to be zeros because they are multiplied with zero, too, but python turns them into nan otherwise
        em[em==0] = 0.000001
        nem[nem==0] = 0.000001
        pnep[pnep==0] = 0.000001
        npep[npep==0] = 0.000001

        ## construct the four coefficients from the SWT
        A = -(np.conj(eta_k)*eta_kp) / em
        B = (np.conj(nu_kp)*nu_k) / nem
        C = (nu_k*eta_kp) / npep
        D = -(np.conj(eta_k)*np.conj(nu_kp)) / pnep
       
        ## calculate energies that appear in the final spin structure

        E1 = A*np.conj(eta_kp)*eta_k * fm - B*np.conj(nu_k)*nu_kp*nnfm + C*np.conj(eta_kp)*np.conj(nu_k)*npfp - D*nu_kp*eta_k*pnfp #k'k
        E2 = -A*np.conj(nu_mkp)*nu_mk*fm + B*np.conj(eta_mk)*eta_mkp*nnfm - C*np.conj(eta_mk)*np.conj(nu_mkp)*npfp + D*nu_mk*eta_mkp*pnfp #-k-k'

        E1_0 = sum(E1 * np.array([1,1,1,1]).reshape(4,1))
        E1_lb = sum(E1 * np.array([1,-1,-1,1]).reshape(4,1))
        E1_l = sum(E1 * np.array([1,1,-1,-1]).reshape(4,1))
        E1_b = sum(E1 * np.array([1,-1,1,-1]).reshape(4,1))

        E2_0 = sum(E2 * np.array([1,1,1,1]).reshape(4,1))
        E2_lb = sum(E2 * np.array([1,-1,-1,1]).reshape(4,1))
        E2_l = sum(E2 * np.array([1,1,-1,-1]).reshape(4,1))
        E2_b = sum(E2 * np.array([1,-1,1,-1]).reshape(4,1))

        ## calculate phases
        phi1 = (k1[:,1] - 1j* k1[:,0])/np.sqrt(k1[:,1]**2 + k1[:,0]**2)
        phi1prime = (kvalues[:,1] - 1j* kvalues[:,0])/np.sqrt(kvalues[:,1]**2 + kvalues[:,0]**2)
        phi2 = np.conj(phi1) * phi1prime
        mplus= phi1prime * phi1
        mminus = np.conj(mplus)

        ## calculate e-function contribution
        one = (kvalues[:,0]-k1[:,0]).reshape((kvalues[:,0]-k1[:,0]).shape[0],1)
        two = (rvec.reshape(rvec.shape[0],1)).T
        position_prefactor = np.exp(1j*np.dot(one,two))
        # position_prefactor = np.zeros(position_prefactor.shape)+1

        # calculate spin structure components and add to result from previous iterations to get sum over entire k-space     
        J_all += 2*np.dot(E1_0, position_prefactor)/(N**2)

        ising = I(mplus, mminus, phi1, phi1prime, phi2)
        I_int1 = ising[:,0]*E1_lb + ising[:,1]*E2_lb + ising[:,2]*E2_0
        I_all += np.dot(I_int1,position_prefactor)/(N**2)
       
        dm = DM(k1, kvalues, phi1, phi1prime, phi2)
        dm = dm[:,0] * E1_l + dm[:,1] *E1_b + dm[:,2]*E2_l + dm[:,3]*E2_b
        D_all += np.dot(dm, position_prefactor)/(N**2)

        xy = Gamma(phi1, phi1prime, phi2, mplus, mminus)
        G_all += np.dot(xy[0]*E1_lb + xy[1]*E2_0, position_prefactor)/(N**2)

    #############################################
    ###### saving results into *.txt files ######
    #############################################

    ## create name details for output
    parameters = [N, mu, gamma, cps_1, cps_3]
    name = ''
    for element in parameters:
        name += '_'+str(np.round(element,3))

    with open('ana_J'+name+'.txt', 'w') as file2:
        for element in J_all:
            # for entry in element:
            file2.write(str(element)+'\n')

    with open('ana_I'+name+'.txt', 'w') as file2:
        for element in I_all:
            for entry in element:
                file2.write(str(entry)+' ')
            file2.write('\n')

    with open('ana_D'+name+'.txt', 'w') as file3:
        for element in D_all:
            for entry in element:
                file3.write(str(entry)+' ')
            file3.write('\n')

    with open('ana_G'+name+'.txt', 'w') as file4:
        # each row is one entry in 3x3 matrix and it is ordered like xx, xy, xz, yx, ...
        for element in G_all:
            file4.write(str(element))
            file4.write('\n')
    