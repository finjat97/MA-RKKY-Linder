import observables as o
import RKKY_diagonalization_selfconsistent as H_diag_self
import time as timer
import numpy as np
import itertools, operator, os
from joblib import dump, load

from tqdm import tqdm 

def spin_loop(parameters, kvalues, spin_pos, iteration):
    # parameters = [sites_x, sites_y, mu, cps, tri, gamma, jott, imp1, imp2]

    positive_kvalues = [kvalues.index(element) for element in kvalues if round(element,6) > 0] #indices of positive k-values
    zero_kvalues = [kvalues.index(element) for element in kvalues if round(element,6) == 0]#indicies of zero k-values
    energies = []

    configurations = [[[0,1/2,0],[0,1/2,0]], [[1/2,0,0],[1/2,0,0]], [[0,-1/2,0],[0,1/2,0]], [[-1/2,0,0],[1/2,0,0]], [[0,1/2,0],[0,-1/2,0]], [[1/2,0,0],[-1/2,0,0]], 
    [[1/2,0,0],[0,1/2,0]], [[0,1/2,0],[1/2,0,0]], [[-1/2,0,0],[0,1/2,0]], [[0,-1/2,0],[1/2,0,0]], [[1/2,0,0],[0,-1/2,0]], [[0,1/2,0],[-1/2,0,0]],
    [[1/2,0,0],[0,0,1/2]], [[1/2,0,0],[0,0,-1/2]], [[-1/2,0,0],[0,0,1/2]], [[-1/2,0,0],[0,0,-1/2]], [[0,1/2,0],[0,0,1/2]], [[0,-1/2,0],[0,0,1/2]],
    [[0,1/2,0],[0,0,-1/2]], [[0,-1/2,0],[0,0,-1/2]], [[0,0,1/2],[0,0,1/2]],  [[0,0,-1/2],[0,0,1/2]],  [[0,0,1/2],[0,0,-1/2]],  [[0,0,-1/2],[0,0,-1/2]]]

    configurations_label = []

    for version in range(len(configurations)):

        version_label = []

        for site in range(len(configurations[version])):
            index = [i for i, element in enumerate(configurations[version][site]) if element != 0][0]
            if index == 0: 
                if configurations[version][site][index] > 0: 
                    version_label += ['→']
                else: version_label += ['←']
            if index == 1:
                if configurations[version][site][index] > 0: version_label += ['x']
                else: version_label += ['.']
            if index == 2:
                if configurations[version][site][index] > 0: version_label += ['↑']
                else: version_label += ['↓']

        configurations_label.append(version_label)
        
    for version in tqdm(range(len(configurations))):
        if iteration:
            eigen, gap = gap_iteration(parameters, kvalues, configurations[version], spin_pos)
        else: eigen = H_diag_self.diagonalize_hamiltonian(parameters, configurations[version], positions=spin_pos)

        energies.append(o.free_energy(eigen, [positive_kvalues, zero_kvalues], output=False))


    return [configurations_label, energies]

def gap_iteration(parameters, kvalues, spin, spin_pos, save=True):
    # parameters = [sites_x, sites_y, mu, cps, tri, gamma, jott, imp1, imp2]

    positive_kvalues = [kvalues.index(element) for element in kvalues if round(element,6) > 0] #indices of positive k-values
    zero_kvalues = [kvalues.index(element) for element in kvalues if round(element,6) == 0]#indicies of zero k-values

    attract_1, attract_3 = parameters[3], parameters[4]

    parameters[3] = [parameters[3]]*parameters[0]*parameters[1]
    parameters[4] = [[parameters[4]]*parameters[0] *parameters[1], [parameters[4]]*parameters[0] *parameters[1]]
    
    start_iteration_time = timer.time()
    iteration = True
    counter = 0 

    while iteration: 
        counter += 1
        coeffis = []

        eigen = H_diag_self.diagonalize_hamiltonian(parameters, spin)
        
        for k in range(parameters[1]):
            coeffis += [H_diag_self.operator_coefficients(eigen,k)] #coeffi[k][u_up, u_down, v_up, v_down][site (N_x)][0,value]]
        gap = o.selfconsistency_gap(eigen, [parameters[3],parameters[4]], coeffis, [positive_kvalues, zero_kvalues]) # gap = [singlet, [triplet up, triplet down]]

        # criterion based on singlet-gap
        difference_imag = [abs(gap[0][site].imag-parameters[3][site].imag) for site in range(len(gap[0]))] # /abs(gap[0][site])
        difference_real = [abs(gap[0][site].real-parameters[3][site].real) for site in range(len(gap[0]))] #/abs(gap[0][site])
        difference_imag_3 = [abs(gap[1][0][site].imag-parameters[4][0][site].imag) for site in range(len(gap[1][0]))]
        difference_real_3 = [abs(gap[1][0][site].real-parameters[4][0][site].real) for site in range(len(gap[1][0]))]

        if max(difference_imag) > 0.0001 or max(difference_real) > 0.0001 or max(difference_imag_3) > 0.0001 or max(difference_real_3) > 0.0001:
            parameters[3] = gap[0]
            parameters[4] = gap[1]

            # print(max(difference_imag),max(difference_real) ,max(difference_imag_3) ,max(difference_real_3))
                
            if counter > 500:
                print('iteration for (singlet, triplet, SOC, RKKY) = ', parameters[3:7], ' stopped d_max ', round(max(difference_real),5), round(max(difference_imag),5) )
                iteration = False
        else: 
            print('iterations gap', counter, 'in t =',round(timer.time()-start_iteration_time,4))
            iteration = False
    
    # start_iteration_3_time = timer.time()
    # counter = 0
    # iteration_3 = True

    # while iteration_3: 
    #     counter += 1
    #     coeffis = []
    #     #print(attract_tri[len(attract_tri)//2], attract[len(attract)//2])
    #     eigen = H_diag_self.diagonalize_hamiltonian(parameters, spin)

    #     for k in range(parameters[1]):
    #         coeffis += [H_diag_self.operator_coefficients(eigen,k)] #coeffi[k][u_up, u_down, v_up, v_down][site (N_x)][0,value]]
    #     gap = o.selfconsistency_gap(eigen, [parameters[3],parameters[4]], coeffis, [positive_kvalues, zero_kvalues])

    #     print(type(attract_tri[3]), (attract_tri[3]))

    #     # criterion based on triplet-gap
    #     difference_imag_3 = [abs(gap[1][0][site].imag-attract_tri[site].imag) for site in range(len(gap[1][0]))]
    #     difference_real_3 = [abs(gap[1][0][site].real-attract_tri[site].real) for site in range(len(gap[1][0]))]

    #     if max(difference_imag_3) > 0.001 or max(difference_real_3) > 0.001:
    #         attract_tri = gap[1]
                
    #         if counter > 500:
    #             print('iteration_3 for (singlet, triplet, SOC, RKKY) = ', parameters[3:7], ' stopped, d_max ', round(max(difference_real_3),5), round(max(difference_imag_3),5) )
    #             iteration_3 = False
    #     else: 
    #         print('iterations tri gap', counter, 'in t =',round(timer.time()-start_iteration_3_time,4))
    #         iteration_3 = False
   
    parameters[3] = attract_1
    parameters[4] = attract_3

    if save:
        name = 'eigen data/eigen_self'
        for element in parameters:
            name += '_'+str(np.round(element,1))
        name += '.txt'
        dump(eigen, name, compress=2)

    return [eigen, gap] #[eigen, [gap singlet, gap triplet]]

def gap_eff(sites_x, gap, tri, step):
    eff, rel = [], []
    gap = [gap[:len(gap)//2], gap[len(gap)//2:]] # gap = [singlet gap, triplet gap], singlet gap = [gap for tri 1, gap for tri 2, ...], gap for tri 1 = [gap of site 0, gap of site 1, ...]
    
    for V in range(len(np.arange(0.5, tri+step, step))):
        # triplet = np.arange(0, tri+step, step)[V]
        for site in range(sites_x):
            # if gap[0][V][site] >= gap[1][V][site]: eff.append((gap[0][V][site]-gap[1][V][site])/(1-triplet))
            # else: eff.append((gap[1][V][site]-gap[0][V][site])/(1-triplet))
            rel.append((gap[0][V][site]- gap[1][V][site]))
    
    rel = [rel[(element-1)*sites_x: element*sites_x] for element in range(len(np.arange(0, tri+step, step)))]

    return eff, rel

def gap_study(parameters, all_kvalues, spin_orientation, spin_positions, iteration=False):
    # parameters = [sites_x, sites_y, mu, cps, tri, gamma, jott, imp1, imp2]

    gap_studies_1, gap_studies_3, density_study, energies, free_energies, labels = [], [], [], [], [], []
    site = parameters[0]//2
    # kvalues = all_kvalues[0]
    step = 0.3
    
    for triplet in np.arange(0.5, parameters[4] + step , step):

        parameters[4] = triplet

        name = 'eigen data/eigen_self'
        for element in parameters:
            name += '_'+str(np.round(element,1))
        name += '.txt'

        if not os.path.exists(name): #check if eigenvalues were already calculated
            if iteration:
                eigen, gap = gap_iteration(parameters, all_kvalues[0], spin_orientation, spin_positions) # gap = [gap_singlet, gap_triplet]
            else: 
                eigen = H_diag_self.diagonalize_hamiltonian(parameters, spin_orientation)
                # positive_kvalues = [kvalues.index(element) for element in kvalues if round(element,6) > 0] #indices of positive k-values
                # zero_kvalues = [kvalues.index(element) for element in kvalues if round(element,6) == 0]#indicies of zero k-values
                gap = [[parameters[3]]*parameters[0]*parameters[1], [parameters[4]]*parameters[0]*parameters[1]]
        
        else:
            eigen = load(name)
            gap = [[parameters[3]]*parameters[0]*parameters[1], [parameters[4]]*parameters[0]*parameters[1]] 

        coeffis = []
        for k in range(parameters[1]):
            coeffis += [H_diag_self.operator_coefficients(eigen,k)] #coeffi[k][u_up, u_down, v_up, v_down][site (N_x)][0,value]] 
        
        #gap_effective = gap_eff(parameters[0], gap, gap[1])
        ldos = o.density_of_states(eigen, coeffis, [site,parameters[0]], [all_kvalues[1], all_kvalues[2]], output=False)
        
        gap_studies_1.append(gap[0]) #singlet gap
        gap_studies_3.append(gap[1]) #triplet gap up
        density_study.append(ldos)
        labels.append(r'$V$ = '+str(round(triplet,2)))
        energies.append(eigen)

        # config, free_energy = spin_loop(parameters, kvalues, parameters[-2:], iteration=False)
        # free_energies.append(free_energy)

        name = 'eigen data/eigen_self'
        for element in parameters:
            name += '_'+str(np.round(element,1))
        name += '.txt'
        dump(eigen, name, compress=2)
    
    return [gap_studies_1+gap_studies_3, energies, density_study, parameters, labels, site, step] #, config, free_energies]

def soc_study(parameters, all_kvalues, spin_orientation, spin_positions, iteration = False):
    # parameters = [sites_x, sites_y, mu, cps, tri, gamma, jott, imp1, imp2]

    gap_studies_1, gap_studies_3, density_study, energies, free_energies, labels, config = [], [], [], [], [], [], []
    site = parameters[0]//2
    kvalues = all_kvalues[0]
    step = 0.2
    
    for soc in (np.arange(0, parameters[5] + step, step)):

        parameters[5] = soc

        if iteration: 
            eigen, gap = gap_iteration(parameters, all_kvalues[0], spin_orientation, spin_positions) # gap = [gap_singlet, gap_triplet]
        else: 
            eigen = H_diag_self.diagonalize_hamiltonian(parameters, spin_orientation)
            # positive_kvalues = [kvalues.index(element) for element in kvalues if round(element,6) > 0] #indices of positive k-values
            # zero_kvalues = [kvalues.index(element) for element in kvalues if round(element,6) == 0]#indicies of zero k-values
            gap = [[parameters[3]]*parameters[0]*parameters[1], [parameters[4]]*parameters[0]*parameters[1]]

        coeffis = []
        for k in range(parameters[1]):
            coeffis += [H_diag_self.operator_coefficients(eigen,k)] #coeffi[k][u_up, u_down, v_up, v_down][site (N_x)][0,value]] 
        
        # if not iteration:
           
        ldos = o.density_of_states(eigen, coeffis, [site,parameters[0]], [all_kvalues[1], all_kvalues[2]], output=False)
        
        gap_studies_1.append(gap[0]) #singlet gap
        gap_studies_3.append(gap[1]) #triplet gap up
        density_study.append(ldos)
        labels.append(r'$\gamma$ = '+str(round(soc,2)))
        energies.append(eigen)

        if parameters[6] != 0: 
            config, free_energy = spin_loop(parameters, kvalues, parameters[-2:], iteration=False)
            free_energies.append(free_energy)
        
        name = 'eigen data/eigen_self'
        for element in parameters:
            name += '_'+str(np.round(element,1))
        name += '.txt'
        dump(eigen, name, compress=2)
    
    return [gap_studies_1+gap_studies_3,  energies, density_study, parameters, labels, site, step, config, free_energies]

def rkky_study(parameters, all_kvalues, spin_orientation, spin_positions, iteration=False):

    # parameters = [sites_x, sites_y, mu, cps, tri, gamma, jott, imp1, imp2]
    
    gap_studies_1, gap_studies_3, density_study_1, density_study_2, energies, labels_1, labels_2 = [], [], [], [],[], [], []
    site = parameters[0]//2
    kvalues = all_kvalues[0]
    step = 0.5
    
    for rkky in (np.arange(0, parameters[6] +step, step)):

        parameters[6] = rkky

        name = 'eigen data/eigen_self'
        for element in parameters:
            name += '_'+str(np.round(element,1))
        name += '.txt'

        # if not os.path.exists(name): #check if eigenvalues were already calculated
        if iteration:
            eigen, gap = gap_iteration(parameters, all_kvalues[0], spin_orientation, spin_positions) # gap = [gap_singlet, gap_triplet]
            print('selfconsistent')
        else: 
            eigen = H_diag_self.diagonalize_hamiltonian(parameters, spin_orientation)
            # positive_kvalues = [kvalues.index(element) for element in kvalues if round(element,6) > 0] #indices of positive k-values
            # zero_kvalues = [kvalues.index(element) for element in kvalues if round(element,6) == 0]#indicies of zero k-values
            gap = [[parameters[3]]*parameters[0]*parameters[1], [parameters[4]]*parameters[0]*parameters[1]]
        
        # else:
        #     print('old eigenvalues')
        #     eigen = load(name)
        #     gap = [[parameters[3]]*parameters[0]*parameters[1], [parameters[4]]*parameters[0]*parameters[1]] 

        coeffis = []
        for k in range(parameters[1]):
            coeffis += [H_diag_self.operator_coefficients(eigen,k)] #coeffi[k][u_up, u_down, v_up, v_down][site (N_x)][0,value]] 
        
        #gap_effective = gap_eff(parameters[0], gap, gap[1])
        ldos_1 = o.density_of_states(eigen, coeffis, [site,parameters[0]], [all_kvalues[1], all_kvalues[2]], output=False)
        #ldos_2 = o.density_of_states(eigen, coeffis, [site+1,parameters[0]], [all_kvalues[1], all_kvalues[2]], output=False)
        
        gap_studies_1.append(gap[0]) #singlet gap
        gap_studies_3.append(gap[1]) #triplet gap up
        density_study_1.append(ldos_1)
        #density_study_2.append(ldos_2)
        labels_1.append(r'$J$ = '+str(round(rkky,2))) #+' at i = '+str(site))
        #labels_2.append(r'$J$ = '+str(round(rkky,2))+' at i = '+str(site+1))
        energies.append(eigen)

        ## save eigenvalues for later purposes
        name = 'eigen data/eigen_self'
        for element in parameters:
            name += '_'+str(np.round(element,1))
        name += '.txt'
        dump(eigen, name, compress=2)
    
    # return [gap_studies, energies + energies, density_study_1 + density_study_2, parameters, labels_1+labels_2, site]
    return [gap_studies_1 + gap_studies_3, energies, density_study_1 , parameters, labels_1, site, step]

def zero_LDOS_gap(parameters, density, eigen, step):
    # find gap by measuring at energy interval of zero-LDOS around zero-energy

    # step = 0.3
    # site = parameters[0]//2
    gap_D = []
    
    for triplet in np.arange(0.5, parameters[4] + step-0.1 , step):
        idx = np.where(np.arange(0.5, parameters[4] + step , step)==triplet)[0][0]
        energies = np.arange(eigen[idx][0][0][0],eigen[idx][0][0][-1]+0.01,0.01)
        interesting_densities = [round(element, 5) for element in density[idx]][[i for i, x in enumerate(energies) if x >0][0]- int((parameters[3]/2)//0.01): [i for i, x in enumerate(energies) if x >0][0]+ int((parameters[3]/2)//0.01) ]
        r = [list(y) for (x,y) in itertools.groupby((enumerate(interesting_densities)),operator.itemgetter(1)) if x == 0]
        if len(r) >= 1:
            r = max(r, key=len )
            energies = energies[[i for i, x in enumerate(energies) if x >0][0]- int((parameters[3]/2)//0.01): [i for i, x in enumerate(energies) if x >0][0]+ int((parameters[3]/2)//0.01) ]
            gap_D.append((r[-1][0]-r[0][0])*0.01)
        else: 
            gap_D.append(0)
    
    return gap_D