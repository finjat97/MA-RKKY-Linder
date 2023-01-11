import numpy as np 

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
    
    density = [] #list of densities for each energy
    energies = np.arange(eigen[0][0][0],eigen[0][0][-1]+0.01,0.01) #list of discretize energy spectrum
    offset = len(eigen[0][0])//2 # makes it easier to jsut choose coefficients belonging to positive eigenvalues
    eigenvalues = range(offset) # gives the indices for all eigenvalue (either positive or negative, depending on if offset is used or not)
          
    for energy in energies:
        sum_1 , sum_2 = 0,0

        for k in kvalues[0]: #all positive k-values

            u_up_eigval = np.square(np.absolute(coefficients[k][0][site][0,np.add(eigenvalues, offset)]))
            u_down_eigval = np.square(np.absolute(coefficients[k][1][site][0,np.add(eigenvalues, offset)]))
            v_up_eigval = np.square(np.absolute(coefficients[k][2][site][0,np.add(eigenvalues, offset)]))
            v_down_eigval = np.square(np.absolute(coefficients[k][3][site][0,np.add(eigenvalues, offset)]))

            deltas_plus =  [delta_function(x) for x in np.add(energy, eigen[k][0][np.add(eigenvalues, offset)])] # delta_function(energies+eigen[k][0][np.add(eigenvalues, offset)])
            deltas_minus = [delta_function(x) for x in np.add(energy, - eigen[k][0][np.add(eigenvalues, offset)])] # delta_function(energies-eigen[k][0][np.add(eigenvalues, offset)])
            
            sum_1 += np.sum(np.multiply(deltas_plus, u_down_eigval + u_up_eigval))
            sum_2 += np.sum(np.multiply(deltas_minus, v_down_eigval + v_up_eigval))

        density.append((sum_1 + sum_2)/sites_x)
    
    if output: 
        print('__density of states__\n', [np.round(element,4) for element in density])
        
    return density

def selfconsistency_gap(eigen, potential, coefficients, kvalues):
    #potential is interaction strength of electrons, format: [singlet potential, triplet potential]
    gap_singlet, gap_triplet_up, gap_triplet_down = [], [],[] #, gap_triplet_left , []

    # going through all sites in x-direction
    for site in range(len(eigen[0][0])//4):
        ## singlet gap
        sum_1 = 0
        # going through all k-values
        for k in kvalues[0]: # k>0
            u_up, u_down, v_up, v_down = coefficients[kvalues[0].index(k)][0][site], coefficients[kvalues[0].index(k)][1][site], coefficients[kvalues[0].index(k)][2][site], coefficients[kvalues[0].index(k)][3][site]
            # going through all eigenvalues
            for value in range(len(eigen[0][0])):
                sum_1 += potential[0][site]*((v_down[0,value]*np.conj(u_up[0,value])-v_up[0,value]*np.conj(u_down[0,value]))*fermi_dis(eigen[k][0][value]) +v_up[0,value]*np.conj(u_down[0,value]))

        for k in kvalues[1]:# k=0
            u_up, u_down, v_up, v_down = coefficients[k][0][site], coefficients[k][1][site], coefficients[k][2][site], coefficients[k][3][site]
            for value in range(len(eigen[0][0])):
                # selecting only positive eigen-energies
                if eigen[k][0][value] > 0: 
                    sum_1 += potential[0][site]*((v_down[0,value]*np.conj(u_up[0,value])-v_up[0,value]*np.conj(u_down[0,value]))*fermi_dis(eigen[k][0][value]) +v_up[0,value]*np.conj(u_down[0,value]))

        ## triplet gap
        sum_3_up_next, sum_3_down_next = 0,0 # , sum_3_pre , 0
        site_next = site + 1 
        #site_pre = site - 1
        #going to the right
        if site_next < len(eigen[0][0])//4:

            # going through all k-values
            for k in kvalues[0]: # k>0
                u_up, u_down, v_up, v_down = coefficients[k][0], coefficients[k][1], coefficients[k][2], coefficients[k][3]
                # going through all eigenvalues
                for value in range(len(eigen[0][0])):
                    sum_3_up_next += potential[1][0][site]*((u_up[site_next][0,value]*np.conj(v_up[site][0,value])-u_down[site][0,value]*np.conj(v_down[site_next][0,value]))*fermi_dis(eigen[k][0][value]) +u_down[site][0,value]*np.conj(v_down[site_next][0,value]))
                    sum_3_down_next += potential[1][1][site]*((u_down[site_next][0,value]*np.conj(v_down[site][0,value])-u_up[site][0,value]*np.conj(v_up[site_next][0,value]))*fermi_dis(eigen[k][0][value]) +u_up[site][0,value]*np.conj(v_up[site_next][0,value]))

            for k in kvalues[1]: # k=0
                u_up, u_down, v_up, v_down = coefficients[k][0], coefficients[k][1], coefficients[k][2], coefficients[k][3]
                for value in range(len(eigen[0][0])):
                    # selecting only positive eigenvalues
                    if eigen[k][0][value] > 0: 
                        sum_3_up_next += potential[1][0][site]*((u_up[site_next][0,value]*np.conj(v_up[site][0,value])-u_down[site][0,value]*np.conj(v_down[site_next][0,value]))*fermi_dis(eigen[k][0][value]) +u_down[site][0,value]*np.conj(v_down[site_next][0,value]))
                        sum_3_down_next += potential[1][1][site]*((u_down[site_next][0,value]*np.conj(v_down[site][0,value])-u_up[site][0,value]*np.conj(v_up[site_next][0,value]))*fermi_dis(eigen[k][0][value]) +u_up[site][0,value]*np.conj(v_up[site_next][0,value]))
            
            if site == 0: first_up, first_down = sum_3_up_next, sum_3_down_next

        else: 
            sum_3_up_next, sum_3_down_next = first_up, first_down
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

        gap_singlet.append(sum_1/(len(eigen[0][0])//4)) # sum divided by N
        gap_triplet_up.append(sum_3_up_next/(len(eigen[0][0])//4))
        gap_triplet_down.append(sum_3_down_next/(len(eigen[0][0])//4))
        #gap_triplet_left.append(sum_3_pre/(len(eigen[0][0])//4))

    return [gap_singlet, [gap_triplet_up, gap_triplet_down]] #, gap_triplet_left

def free_energy(eigen, kvalues, output=False):

    summand1 = 0
    summand2 = 0
    beta = 100 #1/k_b T

    for k in kvalues[0]: #all positive k-values
        # going through all eigenvalues
        for eigenvalue in range(len(eigen[0][0])):
            summand1 += (- eigen[k][0][eigenvalue])/ 2 - (np.log(1+np.exp(-beta*eigen[k][0][eigenvalue])))/beta
    
    for k in kvalues[1]: #all k == 0
        for value in range(len(eigen[0][0])):
            # selecting only positive eigen-energies
            if eigen[k][0][value] > 0: 
                summand2 += -(eigen[k][0][eigenvalue])/2 - (np.log(1+np.exp(-beta*eigen[k][0][eigenvalue])))/beta 
    if output:
        print('__free energy__\n', round(summand1+summand2,3))

    return summand1 + summand2
