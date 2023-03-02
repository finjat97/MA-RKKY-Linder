import numpy as np 
import RKKY_diagonalization_ky as H_diag

def delta_function(x):
    # Gaussian approximation
    width= 0.05

    return 1/(width*np.sqrt(2*np.pi))*np.exp((-1/2) * (x**2/width**2))

def fermi_dis(energy):
    # assumed to be the expectation value <c^dag c>
    constant = 0.01 # k_B*T

    return 1/ (np.exp(energy/constant)+1)

def density_of_states(eigen, coefficients, position, kvalues, output=False):

    site = position[0]
    sites_x  = position[1]
    den = []
    
    energies = np.arange(np.min(eigen[:,:,0]).real,np.max(eigen[:,:,0]).real+0.01,0.01) #list of discretize energy spectrum
    offset = len(eigen[0][0])//2 # makes it easier to just choose coefficients belonging to positive eigenvalues
    eigenvalues = range(offset) # gives the indices for all eigenvalues (either positive or negative, depending on if offset is used or not)
    # def uu(x):
    #     return np.square(np.absolute(coefficients[x][0][site][0,np.add(eigenvalues, offset)]))
    # def ud(x):
    #     return np.square(np.absolute(coefficients[x][1][site][0,np.add(eigenvalues, offset)]))
    # def vu(x):
    #     return np.square(np.absolute(coefficients[x][2][site][0,np.add(eigenvalues, offset)]))
    # def vd(x):
    #     return np.square(np.absolute(coefficients[x][3][site][0,np.add(eigenvalues, offset)]))
    # def dm(y):
    #     return [delta_function(x) for x in np.add(energy, - eigen[y][0][np.add(eigenvalues, offset)])] 
    # def dp(y):
    #     return [delta_function(x) for x in np.add(energy, eigen[y][0][np.add(eigenvalues, offset)])] 


    u_up, u_down, v_up, v_down = coefficients
    u_up = np.absolute(np.square(u_up[:,:,site]) )
    u_down = np.absolute(np.square(u_down[:,:,site] ))
    v_up = np.absolute(np.square(v_up[:,:,site] ))
    v_down = np.absolute(np.square(v_down[:,:,site] ))
     
    eigenenergies = eigen[:,:sites_x*2,0] #find it, make it format shape = (k,n) or (k,n,1)
    us = u_up + u_down
    vs = v_up + v_down

   
    #calculate deltas for different energies

    def d_E(E):
        density = sum(sum(us* delta_function(E+eigenenergies) + vs * delta_function(E-eigenenergies)))
        return density/sites_x

    for energy in energies:
        den.append(d_E(energy))

    # for energy in energies:
    #     sum_1 , sum_2 = 0,0
    #     # calculate coefficients for all positive k values
    #     u_up_eigval = list(map(uu, kvalues[0]))
    #     u_down_eigval = list(map(ud, kvalues[0]))
    #     v_up_eigval = list(map(vu, kvalues[0]))
    #     v_down_eigval = list(map(vd, kvalues[0]))
    #     # calculate delta functions, which are prefactors, for all positive k values
    #     deltas_plus = list(map(dp, kvalues[0]))
    #     deltas_minus = list(map(dm, kvalues[0]))
    #     # multiply the two different terms and add each one together, see numerical approach in master thesis for further explanation
    #     sum_1 += np.sum(np.multiply(deltas_plus, u_down_eigval + u_up_eigval))
    #     sum_2 += np.sum(np.multiply(deltas_minus, v_down_eigval + v_up_eigval))

    #     density.append((sum_1 + sum_2)/sites_x)
    # print(energies.shape)
    # res = d_E(energies)
    # print(res)
    # # if output: 
    #     print('__density of states__\n', [np.round(element,4) for element in density])

    return den, energies

def selfconsistency_gap(eigen, potential, coefficients, kvalues):
    #potential is interaction strength of electrons, format: [singlet potential, triplet potential]
    gap_singlet, gap_triplet_up, gap_triplet_down, gap_triplet, gap_triplet_ud, gap_triplet_du = [], [], [], [], [], [] #, gap_triplet_left , []

    # going through all sites in x-direction
    for site in range(len(eigen[0][0])//4):
        ## singlet gap
        sum_1 = 0
        # going through all k-values
        for k in kvalues[0]: # k>0
            u_up, u_down, v_up, v_down = coefficients[kvalues[0].index(k)][0][site], coefficients[kvalues[0].index(k)][1][site], coefficients[kvalues[0].index(k)][2][site], coefficients[kvalues[0].index(k)][3][site]
            # going through all eigenvalues
            for value in range(len(eigen[0][0])):
                sum_1 -= potential[0][site]*((v_down[0,value]*np.conj(u_up[0,value])-v_up[0,value]*np.conj(u_down[0,value]))*fermi_dis(eigen[k][0][value]) +v_up[0,value]*np.conj(u_down[0,value]))
        # print('1', sum_1)
        for k in kvalues[1]:# k=0
            u_up, u_down, v_up, v_down = coefficients[k][0][site], coefficients[k][1][site], coefficients[k][2][site], coefficients[k][3][site]
            for value in range(len(eigen[0][0])):
                # selecting only positive eigen-energies
                if eigen[k][0][value] > 0: 
                    sum_1 -= potential[0][site]*((v_down[0,value]*np.conj(u_up[0,value])-v_up[0,value]*np.conj(u_down[0,value]))*fermi_dis(eigen[k][0][value]) +v_up[0,value]*np.conj(u_down[0,value]))
        # print('1.2',sum_1)
        ## triplet gap
        sum_3_up_next, sum_3_down_next, sum_3_updown_next, sum_3_downup_next = 0,0,0,0 # , sum_3_pre , 0
        sum_3 = 0
        site_next = site + 1 
        #site_pre = site - 1
        #going to the right
        if site_next < len(eigen[0][0])//4:

            # going through all k-values
            for k in kvalues[0]: # k>0
                u_up, u_down, v_up, v_down = coefficients[k][0], coefficients[k][1], coefficients[k][2], coefficients[k][3]
                # going through all eigenvalues
                for value in range(len(eigen[0][0])):
                    #
                    sum_3_up_next -= potential[1][0][site]*((u_up[site_next][0,value]*np.conj(v_up[site][0,value])-u_down[site][0,value]*np.conj(v_down[site_next][0,value]))*fermi_dis(eigen[k][0][value]) +u_down[site][0,value]*np.conj(v_down[site_next][0,value]))
                    sum_3_down_next -= potential[1][1][site]*((u_down[site_next][0,value]*np.conj(v_down[site][0,value])-u_up[site][0,value]*np.conj(v_up[site_next][0,value]))*fermi_dis(eigen[k][0][value]) +u_up[site][0,value]*np.conj(v_up[site_next][0,value]))
                    sum_3_updown_next -= potential[1][0][site]*((u_up[site_next][0,value]*np.conj(v_down[site][0,value])-u_down[site][0,value]*np.conj(v_up[site_next][0,value]))*fermi_dis(eigen[k][0][value]) +u_down[site][0,value]*np.conj(v_up[site_next][0,value]))
                    sum_3_downup_next -= potential[1][1][site]*((u_down[site_next][0,value]*np.conj(v_up[site][0,value])-u_up[site][0,value]*np.conj(v_down[site_next][0,value]))*fermi_dis(eigen[k][0][value]) +u_up[site][0,value]*np.conj(v_down[site_next][0,value]))
            # print('3', sum_3_down_next + sum_3_up_next + sum_3_downup_next + sum_3_updown_next)
            for k in kvalues[1]: # k=0
                u_up, u_down, v_up, v_down = coefficients[k][0], coefficients[k][1], coefficients[k][2], coefficients[k][3]
                for value in range(len(eigen[0][0])):
                    # selecting only positive eigenvalues
                    if eigen[k][0][value] > 0: 
                        sum_3_up_next -= potential[1][0][site]*((u_up[site_next][0,value]*np.conj(v_up[site][0,value])-u_down[site][0,value]*np.conj(v_down[site_next][0,value]))*fermi_dis(eigen[k][0][value]) +u_down[site][0,value]*np.conj(v_down[site_next][0,value]))
                        sum_3_down_next -= potential[1][1][site]*((u_down[site_next][0,value]*np.conj(v_down[site][0,value])-u_up[site][0,value]*np.conj(v_up[site_next][0,value]))*fermi_dis(eigen[k][0][value]) +u_up[site][0,value]*np.conj(v_up[site_next][0,value]))
                        sum_3_updown_next -= potential[1][0][site]*((u_up[site_next][0,value]*np.conj(v_down[site][0,value])-u_down[site][0,value]*np.conj(v_up[site_next][0,value]))*fermi_dis(eigen[k][0][value]) +u_down[site][0,value]*np.conj(v_up[site_next][0,value]))
                        sum_3_downup_next -= potential[1][1][site]*((u_down[site_next][0,value]*np.conj(v_up[site][0,value])-u_up[site][0,value]*np.conj(v_down[site_next][0,value]))*fermi_dis(eigen[k][0][value]) +u_up[site][0,value]*np.conj(v_down[site_next][0,value]))
            # print('3.2', sum_3_down_next + sum_3_up_next + sum_3_downup_next + sum_3_updown_next)
            #if site == 0: first_up, first_down = sum_3_up_next, sum_3_down_next
            if site == 1: first = sum_3

        else: 
            sum_3 = first
        #going to the left
        # if site_pre > 0:

        #     # going through all k-values
        #     for k in kvalues[0]: # k!=0
        #         u_up, u_down, v_up, v_down = coefficients[k][0], coefficients[k][1], coefficients[k][2], coefficients[k][3]
        #         # going through all eigenvalues
        #         for value in range(len(eigen[0][0])):
        #             sum_3_pre += potential[1]*((u_up[site_pre][0,value]*np.conj(v_up[site][0,value])-u_down[site][0,value]*np.conj(v_down[site_pre][0,value]))*fermi_dis(eigen[k][0][value]) +u_down[site][0,value]*np.conj(v_down[site_pre][0,value]))
        #             sum_3_pre += potential[1]*((u_down[site_pre][0,value]*np.conj(v_down[site][0,value])-u_up[site][0,value]*np.conj(v_up[site_pre][0,value]))*fermi_dis(eigen[k][0][value]) +u_up[site][0,value]*np.conj(v_up[site_pre][0,value]))

        #     for k in kvalues[1]: # k=0
        #         u_up, u_down, v_up, v_down = coefficients[k][0], coefficients[k][1], coefficients[k][2], coefficients[k][3]
        #         for value in range(len(eigen[0][0])):
        #             # selecting only positive eigenvalues
        #             if eigen[k][0][value] > 0: 
        #                 sum_3_pre += potential[1]*((u_up[site_pre][0,value]*np.conj(v_up[site][0,value])-u_down[site][0,value]*np.conj(v_down[site_pre][0,value]))*fermi_dis(eigen[k][0][value]) +u_down[site][0,value]*np.conj(v_down[site_pre][0,value]))
        #                 sum_3_pre += potential[1]*((u_down[site_pre][0,value]*np.conj(v_down[site][0,value])-u_up[site][0,value]*np.conj(v_up[site_pre][0,value]))*fermi_dis(eigen[k][0][value]) +u_up[site][0,value]*np.conj(v_up[site_pre][0,value]))
        
        # else: 
        #     sum_3_pre = 0
        sum_3 = sum_3_down_next + sum_3_up_next + sum_3_downup_next + sum_3_updown_next
        gap_singlet.append(sum_1/(len(eigen[0][0])//4)) # sum divided by N
        gap_triplet_up.append(sum_3_up_next/(len(eigen[0][0])//4))
        gap_triplet_down.append(sum_3_down_next/(len(eigen[0][0])//4))
        gap_triplet_ud.append((sum_3_updown_next)/(len(eigen[0][0])//4))
        gap_triplet_du.append((sum_3_downup_next)/(len(eigen[0][0])//4))
        gap_triplet.append(sum_3/(len(eigen[0][0])//4))
        #gap_triplet_left.append(sum_3_pre/(len(eigen[0][0])//4))

    return [gap_singlet, [gap_triplet_up, gap_triplet_down, gap_triplet_ud, gap_triplet_du], gap_triplet] #, gap_triplet_left

def free_energy(eigen, kvalues, output=False):
    beta = 100 #1/k_b T
    eigvals = eigen[:,:,0]
    # print(eigvals.shape)
    eigvals = eigvals[eigvals.shape[0]//2:,:]
    # print(eigvals.shape)
    #print(kvalues[0])
    # eigvals_posk = eigvals[kvalues[0]]
    # sum_posk = sum( sum ( -1/2 * eigvals_posk - 1/beta * np.log( 1+ np.exp(-beta*eigvals_posk)) ))
    # eigvals_zerok = eigvals[kvalues[1]]

    # sum_zerok = sum (sum( -1/2 * eigvals_zerok - 1/beta * np.log( 1+ np.exp(-beta*eigvals_zerok)) ) )

    # print(sum_zerok)
    F = sum( sum ( -1/2 * eigvals - 1/beta * np.log( 1+ np.exp(-beta*eigvals)) ))
    
    # # find all combinations of k values and eigenvalue indices
    # args_p = np.array(np.meshgrid(kvalues[0], list(range(len(eigen[0][0]))))).T.reshape(-1,2)
    # args_0 = np.array(np.meshgrid(kvalues[1], list(range(len(eigen[0][0]))))).T.reshape(-1,2)
    # # make sure to only consider positive energies
    # args_0 = [item for item in args_0 if eigen[args_0[0][0]][0][args_0[0][1]]>0]
    # # how to calculate the energy
    # def energy(x):
    #     return (- eigen[x[0]][0][x[1]])/ 2 - (np.log(1+np.exp(-beta*eigen[x[0]][0][x[1]])))/beta
    # # calculate free energy
    # free_energy = sum(map(energy, args_p)) + sum(map(energy, args_0))

    # for k in kvalues[0]: #all positive k-values
    #     # going through all eigenvalues
    #     for eigenvalue in range(len(eigen[0][0])):
    #         summand1 += (- eigen[k][0][eigenvalue])/ 2 - (np.log(1+np.exp(-beta*eigen[k][0][eigenvalue])))/beta
    
    # for k in kvalues[1]: #all k == 0
    #     for value in range(len(eigen[0][0])):
    #         # selecting only positive eigen-energies
    #         if eigen[k][0][value] > 0: 
    #             summand2 += -(eigen[k][0][eigenvalue])/2 - (np.log(1+np.exp(-beta*eigen[k][0][eigenvalue])))/beta 
    if output:
        print('__free energy__\n', round(free_energy,3))

    return F

def interband_pairing(eigen, kvalues, coeffis):

    for site in range(len(eigen[0][0])//4):
        ## triplet gap
        sum_3_up_next, sum_3_down_next = 0,0 # , sum_3_pre , 0
        site_next = site + 1 
        #site_pre = site - 1
        #going to the right
        if site_next < len(eigen[0][0])//4:

            # going through all k-values
            for k in kvalues[0]: # k>0
                u_up, u_down, v_up, v_down = coeffis[k][0], coeffis[k][1], coeffis[k][2], coeffis[k][3]
                # going through all eigenvalues
                for value in range(len(eigen[0][0])):
                    sum_3_up_next += ((u_up[site_next][0,value]*np.conj(v_up[site][0,value])-u_down[site][0,value]*np.conj(v_down[site_next][0,value]))*fermi_dis(eigen[k][0][value]) +u_down[site][0,value]*np.conj(v_down[site_next][0,value]))
                    sum_3_down_next += ((u_down[site_next][0,value]*np.conj(v_down[site][0,value])-u_up[site][0,value]*np.conj(v_up[site_next][0,value]))*fermi_dis(eigen[k][0][value]) +u_up[site][0,value]*np.conj(v_up[site_next][0,value]))

            for k in kvalues[1]: # k=0
                u_up, u_down, v_up, v_down = coeffis[k][0], coeffis[k][1], coeffis[k][2], coeffis[k][3]
                for value in range(len(eigen[0][0])):
                    # selecting only positive eigenvalues
                    if eigen[k][0][value] > 0: 
                        sum_3_up_next += ((u_up[site_next][0,value]*np.conj(v_up[site][0,value])-u_down[site][0,value]*np.conj(v_down[site_next][0,value]))*fermi_dis(eigen[k][0][value]) +u_down[site][0,value]*np.conj(v_down[site_next][0,value]))
                        sum_3_down_next += ((u_down[site_next][0,value]*np.conj(v_down[site][0,value])-u_up[site][0,value]*np.conj(v_up[site_next][0,value]))*fermi_dis(eigen[k][0][value]) +u_up[site][0,value]*np.conj(v_up[site_next][0,value]))
            
            if site == 0: first_up, first_down = sum_3_up_next, sum_3_down_next

        else: 
            sum_3_up_next, sum_3_down_next = first_up, first_down

        amplitude = sum_3_up_next/(len(eigen[0][0])//4)-sum_3_down_next/(len(eigen[0][0])//4)

    return amplitude