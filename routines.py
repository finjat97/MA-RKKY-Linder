import observables as o
import RKKY_diagonalization_ky as H_diag
import time as timer
import numpy as np
import itertools, operator, os
from joblib import dump, load
import spinstructure_numbers as analytical

from tqdm import tqdm 

def spin_loop(parameters, kvalues, spin_pos, iteration=False):
    # parameters = [sites_x, sites_y, mu, cps, tri, gamma, jott, imp1, imp2]

    spin = 1/2

    def spin_orientation(winkel):
        res = np.array([np.round(spin*np.sin(winkel[:,0])*np.cos(winkel[:,1]),5), np.round(spin*np.sin(winkel[:,0])*np.sin(winkel[:,1]),5), np.round(spin*np.cos(winkel[:,0]),5)])
        return res +0

    step = 4 # number of intervals to dicretisize the spherical coordinates
    # find all angle combinations for discretisized spherical coordinates
    angle = np.array(np.meshgrid(np.arange(-np.pi, np.pi+np.pi/step, np.pi/step), np.arange(-np.pi, np.pi+np.pi/step, np.pi/step))).T.reshape(-1,2)
    # calculate all possible orientations based on those angle for one spin
    one_spin = spin_orientation(angle).T
    one_spin = np.unique(one_spin, axis=0) #select only spins that are unique
    # find all possible combinations of two arrays with length of the angle combinations (aka. number of possible directions of one spin)
    combo = np.array(np.meshgrid(range(one_spin.shape[0]),range(one_spin.shape[0]))).T.reshape(-1,2)
    # find all possible ways to combine two spins with all directions allowed by the previous calculated angle combinations
    two_spin = np.array(one_spin[combo]) #shape = (possible directions **2, 2, 3)   

    positive_kvalues = [kvalues.index(element) for element in kvalues if round(element,6) > 0] #indices of positive k-values
    zero_kvalues = [kvalues.index(element) for element in kvalues if round(element,6) == 0]#indicies of zero k-values
    energies = []
        
    for spin1, spin2 in two_spin:
        if iteration:
            eigen, gap = gap_iteration(parameters, kvalues, [spin1, spin2], spin_pos)
        else: 
            eigen = H_diag.diagonalize_hamiltonian(parameters, [spin1, spin2], positions=spin_pos)

        energies.append(o.free_energy(eigen, [positive_kvalues, zero_kvalues], output=False))


    return [two_spin, energies]

def gap_iteration(parameters, kvalues, spin, spin_pos, save=True):
    # parameters = [sites_x, sites_y, mu, cps, tri, gamma, jott, imp1, imp2]

    positive_kvalues = [kvalues.index(element) for element in kvalues if round(element,6) > 0] #indices of positive k-values
    zero_kvalues = [kvalues.index(element) for element in kvalues if round(element,6) == 0]#indicies of zero k-values

    attract = [parameters[3]/2]*parameters[0]*parameters[1]
    attract_tri = [parameters[4]/2]*parameters[0] *parameters[1]
    
    start_iteration_time = timer.time()
    iteration = True
    counter = 0 

    while iteration: 
        counter += 1
        coeffis = []

        eigen = H_diag.diagonalize_hamiltonian(parameters[0], parameters[1], 1, parameters[2], parameters[5], attract, attract_tri, parameters[6] ,spin, spin_pos, output=False) # list with N_x entries, entry = (eigvals, eigvecs) of k
        
        def H_coeffi(x):
            return H_diag.operator_coefficients(eigen,x)
        coeffis = list(map(H_coeffi, list(range(parameters[1]))))
        # for k in range(parameters[1]):
        #     coeffis += [H_diag.operator_coefficients(eigen,k)] #coeffi[k][u_up, u_down, v_up, v_down][site (N_x)][0,value]]
        gap = o.selfconsistency_gap(eigen, [parameters[3],parameters[4]], coeffis, [positive_kvalues, zero_kvalues]) # gap = [singlet, [triplet up, triplet down]]

        # criterion based on singlet-gap
        difference_imag = [abs(gap[0][site].imag-attract[site].imag) for site in range(len(gap[0]))] # /abs(gap[0][site])
        difference_real = [abs(gap[0][site].real-attract[site].real) for site in range(len(gap[0]))] #/abs(gap[0][site])

        if max(difference_imag) > 0.001 or max(difference_real) > 0.001:
            attract = gap[0]
                
            if counter > 500:
                print('iteration for (singlet, triplet, SOC, RKKY) = ', parameters[3:7], ' stopped d_max ', round(max(difference_real),5), round(max(difference_imag),5) )
                iteration = False
        else: 
            print('iterations gap', counter, 'in t =',round(timer.time()-start_iteration_time,4))
            iteration = False
    
    start_iteration_3_time = timer.time()
    counter = 0
    iteration_3 = True

    while iteration_3: 
        counter += 1
        coeffis = []
        #print(attract_tri[len(attract_tri)//2], attract[len(attract)//2])
        eigen = H_diag.diagonalize_hamiltonian(parameters[0], parameters[1], 1, parameters[2], parameters[5], attract, attract_tri, parameters[6] ,spin, spin_pos, output=False) # list with N_x entries, entry = (eigvals, eigvecs) of k
        
        for k in range(parameters[1]):
            coeffis += [H_diag.operator_coefficients(eigen,k)] #coeffi[k][u_up, u_down, v_up, v_down][site (N_x)][0,value]]
        gap = o.selfconsistency_gap(eigen, [parameters[3],parameters[4]], coeffis, [positive_kvalues, zero_kvalues])

        # criterion based on triplet-gap
        difference_imag_3 = [abs(gap[1][site].imag-attract_tri[site].imag) for site in range(len(gap[1]))]
        difference_real_3 = [abs(gap[1][site].real-attract_tri[site].real) for site in range(len(gap[1]))]

        if max(difference_imag_3) > 0.001 or max(difference_real_3) > 0.001:
            attract_tri = gap[1]
                
            if counter > 500:
                print('iteration_3 for (singlet, triplet, SOC, RKKY) = ', parameters[3:7], ' stopped, d_max ', round(max(difference_real_3),5), round(max(difference_imag_3),5) )
                iteration_3 = False
        else: 
            print('iterations tri gap', counter, 'in t =',round(timer.time()-start_iteration_3_time,4))
            iteration_3 = False
   
    if save:
        name = 'eigen data/eigen'
        for element in parameters:
            name += '_'+str(np.round(element,3))
        name += '.txt'
        dump(eigen, name, compress=2)

    return [eigen, gap] #[eigen, [gap singlet, gap triplet]]

def gap_eff(sites_x, gap, tri, step):
    eff, rel = [], []
    gap = [gap[:len(gap)//2], gap[len(gap)//2:]] # gap = [singlet gap, triplet gap], singlet gap = [gap for tri 1, gap for tri 2, ...], gap for tri 1 = [gap of site 0, gap of site 1, ...]
    
    for V in range(len(np.arange(0.5, tri+step, step))):
        # triplet = np.arange(0, tri+step, step)[V]
        def rel_gap(x):
            return (gap[0][V][x]- gap[1][V][x])
        rel = list(map(rel_gap, list(range(sites_x))))
        # for site in range(sites_x):
        #     # if gap[0][V][site] >= gap[1][V][site]: eff.append((gap[0][V][site]-gap[1][V][site])/(1-triplet))
        #     # else: eff.append((gap[1][V][site]-gap[0][V][site])/(1-triplet))
        #     rel.append((gap[0][V][site]- gap[1][V][site]))
    
    rel = [rel[(element-1)*sites_x: element*sites_x] for element in range(len(np.arange(0, tri+step, step)))]

    return eff, rel

def gap_study(parameters, all_kvalues, spin_orientation, spin_positions, iteration=False):
    # parameters = [sites_x, sites_y, mu, cps, tri, gamma, jott, imp1, imp2]

    # gap_studies_1, gap_studies_3, density_study, energies, free_energies, labels = [], [], [], [], [], []
    gap_studies_1, gap_studies_3, density_study, density_study_2, density_study_3, energies, free_energies, labels, labels_2, labels_3 = [], [], [], [], [], [], [], [],[],[]

    site = parameters[0]//2 -1
    # kvalues = all_kvalues[0]
    step = 0.1
    
    for triplet in np.arange(0.3, parameters[4] + step , step):

        parameters[4] = triplet

        name = 'eigen data/eigen'
        for element in parameters:
            name += '_'+str(np.round(element,3))
        name += '.txt'

        if not os.path.exists(name): #check if eigenvalues were already calculated
            if iteration:
                eigen, gap = gap_iteration(parameters, all_kvalues[0], spin_orientation, spin_positions) # gap = [gap_singlet, gap_triplet]
            else: 
                eigen = H_diag.diagonalize_hamiltonian(parameters, spin_orientation)
                # positive_kvalues = [kvalues.index(element) for element in kvalues if round(element,6) > 0] #indices of positive k-values
                # zero_kvalues = [kvalues.index(element) for element in kvalues if round(element,6) == 0]#indicies of zero k-values
                gap = [[parameters[3]]*parameters[0]*parameters[1], [parameters[4]]*parameters[0]*parameters[1]]
        
        else:
            eigen = load(name)
            gap = [[parameters[3]]*parameters[0]*parameters[1], [parameters[4]]*parameters[0]*parameters[1]] 

        # def H_coeffi(x):
        #     return H_diag.operator_coefficients(eigen,x)
        coeffis = H_diag.operator_coefficients(eigen, 6)

        # coeffis = []
        # for k in range(parameters[1]):
        #     coeffis += [H_diag.operator_coefficients(eigen,k)] #coeffi[k][u_up, u_down, v_up, v_down][site (N_x)][0,value]] 
        
        #gap_effective = gap_eff(parameters[0], gap, gap[1])
        ldos = o.density_of_states(eigen, coeffis, [site,parameters[0]], [all_kvalues[1], all_kvalues[2]], output=False)
        # ldos_2 = o.density_of_states(eigen, coeffis, [45,parameters[0]], [all_kvalues[1], all_kvalues[2]], output=False)
        # ldos_3 = o.density_of_states(eigen, coeffis, [55,parameters[0]], [all_kvalues[1], all_kvalues[2]], output=False)
        
        
        gap_studies_1.append(gap[0]) #singlet gap
        gap_studies_3.append(gap[1]) #triplet gap up
        density_study.append(ldos[0])
        # density_study_2.append(ldos_2)
        # density_study_3.append(ldos_3)
        labels.append(r'$V$ = '+str(round(triplet,3)))
        # labels_2.append('i = '+str(45))
        # labels_3.append('i = '+str(55))
        energies.append(ldos[1])

        # config, free_energy = spin_loop(parameters, kvalues, parameters[-2:], iteration=False)
        # free_energies.append(free_energy)

        name = 'eigen data/eigen'
        for element in parameters:
            name += '_'+str(np.round(element,3))
        name += '.txt'
        dump(eigen, name, compress=2)
    
    # return [gap_studies_1+gap_studies_3, 3*energies, density_study+density_study_2+density_study_3, parameters, labels+labels_2+labels_3, site, step] #, config, free_energies]
    return [gap_studies_1+gap_studies_3, energies, density_study, parameters, labels, site, step]

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
            eigen = H_diag.diagonalize_hamiltonian(parameters, spin_orientation)
        
        coeffis = H_diag.operator_coefficients(eigen, 6) #six is just placeholder for possible k dependence

        # for k in range(parameters[1]):
        #     coeffis += [H_diag.operator_coefficients(eigen,k)] #coeffi[k][u_up, u_down, v_up, v_down][site (N_x)][0,value]] 
        
        if not iteration:
            gap = [[parameters[3]]*parameters[0]*parameters[1], [parameters[4]]*parameters[0]*parameters[1]]
           
        ldos = o.density_of_states(eigen, coeffis, [site,parameters[0]], [all_kvalues[1], all_kvalues[2]], output=False)
        
        gap_studies_1.append(gap[0]) #singlet gap
        gap_studies_3.append(gap[1]) #triplet gap up
        density_study.append(ldos[0])
        labels.append(r'$\gamma$ = '+str(round(soc,2)))
        energies.append(ldos[1])

        # if parameters[6] != 0: 
        #     config, free_energy = spin_loop(parameters, kvalues, parameters[-2:], iteration=False)
        #     free_energies.append(free_energy)
        
        name = 'eigen data/eigen'
        for element in parameters:
            name += '_'+str(np.round(element,3))
        name += '.txt'
        dump(eigen, name, compress=2)

    dump(free_energies, 'soc_F.txt', compress=2)
    
    return [gap_studies_1+gap_studies_3,  energies, density_study, parameters, labels, site, step, config, free_energies]

def rkky_study(parameters, all_kvalues, spin_orientation, spin_positions, iteration=False):

    # parameters = [sites_x, sites_y, mu, cps, tri, gamma, jott, imp1, imp2]
    
    # gap_studies_1, gap_studies_3, density_study_1, density_study_2, density_study_3, energies, labels_1, labels_2, labels_3 = [], [], [], [],[], [], [], [], []
    gap_studies_1, gap_studies_3, density_study_1, energies, labels_1= [], [], [], [], []
    site = parameters[0]//2 -1
    kvalues = all_kvalues[0]
    step = 1
    
    for rkky in (np.arange(0, parameters[6] +step, step)):

        parameters[6] = rkky

        name = 'eigen data/eigen'
        for element in parameters:
            name += '_'+str(np.round(element,3))
        name += '.txt'

        if not os.path.exists(name): #check if eigenvalues were already calculated
            if iteration:
                eigen, gap = gap_iteration(parameters, all_kvalues[0], spin_orientation, spin_positions) # gap = [gap_singlet, gap_triplet]
            else: 
                eigen = H_diag.diagonalize_hamiltonian(parameters, spin_orientation)
                # positive_kvalues = [kvalues.index(element) for element in kvalues if round(element,6) > 0] #indices of positive k-values
                # zero_kvalues = [kvalues.index(element) for element in kvalues if round(element,6) == 0]#indicies of zero k-values
                gap = [[parameters[3]]*parameters[0]*parameters[1], [parameters[4]]*parameters[0]*parameters[1]]
        
        else:
            eigen = load(name)
            gap = [[parameters[3]]*parameters[0]*parameters[1], [parameters[4]]*parameters[0]*parameters[1]] 

        # def H_coeffi(x):
        #     return H_diag.operator_coefficients(eigen,x)
        # coeffis = list(map(H_coeffi, list(range(parameters[1]))))
        coeffis = H_diag.operator_coefficients(eigen, 6)
        # coeffis = []
        # for k in range(parameters[1]):
        #     coeffis += [H_diag.operator_coefficients(eigen,k)] #coeffi[k][u_up, u_down, v_up, v_down][site (N_x)][0,value]] 
        
        #gap_effective = gap_eff(parameters[0], gap, gap[1])
        ldos_1 = o.density_of_states(eigen, coeffis, [site,parameters[0]], [all_kvalues[1], all_kvalues[2]], output=False)
        # ldos_2 = o.density_of_states(eigen, coeffis, [30,parameters[0]], [all_kvalues[1], all_kvalues[2]], output=False)
        # ldos_3 = o.density_of_states(eigen, coeffis, [25,parameters[0]], [all_kvalues[1], all_kvalues[2]], output=False)
        
        gap_studies_1.append(gap[0]) #singlet gap
        gap_studies_3.append(gap[1]) #triplet gap up
        density_study_1.append(ldos_1)
        # density_study_2.append(ldos_2)
        # density_study_3.append(ldos_3)
        labels_1.append(r'$J$ = '+str(round(rkky,2))+' at i = '+str(site))
        # labels_2.append(r'$J$ = '+str(round(rkky,2))+' at i = '+str(30))
        # labels_3.append(r'$J$ = '+str(round(rkky,2))+' at i = '+str(25))
        energies.append(eigen)

        ## save eigenvalues for later purposes
        name = 'eigen data/eigen'
        for element in parameters:
            name += '_'+str(np.round(element,3))
        name += '.txt'
        dump(eigen, name, compress=2)
    
    
    return [gap_studies_1 + gap_studies_3, energies, density_study_1 , parameters, labels_1, site, step]
    # [gap_studies_1 + gap_studies_3, energies + energies+energies, density_study_1 + density_study_2 + density_study_3, parameters, labels_1+labels_2+labels_3, site]#

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

def distance(parameters, positive_kvalues, zero_kvalues, index):

    # define spin orientation
    spin = [[[0,1/2,0],[0,1/2,0]],[[0,1/2,0],[0,-1/2,0]]]
    F_difference, F_difference_an = [], []
    #create list with different parameter configurations, changing the parameter given with function call
    values = np.arange(0,1,0.2)

    for value in tqdm(values):
        parameters[index] = value
        F_upup, F_updown= [], []
        for pos2 in np.arange(15,parameters[0]-15): #move second impurity away from first by going throught all lattice sites except edge
            # calculate free energy for parallel spins in +y direction
            eigen_upup = H_diag.diagonalize_hamiltonian(parameters, spin[0], positions=[14,pos2])
            F_upup.append(o.free_energy(eigen_upup, [positive_kvalues, zero_kvalues], output=False))
            ## calculate free energy for anti-parallel spins in +-y direction
            eigen_updown = H_diag.diagonalize_hamiltonian(parameters, spin[1], positions=[4,pos2])
            F_updown.append(o.free_energy(eigen_updown, [positive_kvalues, zero_kvalues], output=False))
        # calculate difference in free energy for parallel and anti-parallel orientation
        F_difference += [list(map(lambda x,y: x-y , F_upup, F_updown))]
    # save separation distances of impurities for later plotting
    distances = np.arange(15,parameters[0]-15)-14

    name = 'dF/dF_dis'
    for element in parameters:
        name += '_'+str(np.round(element,2))
    name += '.png'

    dump(distances, name, compress=2)
    name = 'dF/dF_dF'
    for element in parameters:
        name += '_'+str(np.round(element,2))
    name += '.png'
    dump(F_difference, name, compress=2)

    # calculate F difference from analytical expression
    # inter_results = np.array(list(map(analytical.main, [80]*len(distances), [1]*len(distances), [0.1]*len(distances), [2]*len(distances), [1]*len(distances), [0.04]*len(distances) , [0.01]*len(distances), distances)))
    # inter_results = analytical.main(31, 1, 0.1, 2, 1, 0.04, 0.01, distances)
    # print(inter_results.shape)
    # print(inter_results[1])
    
    # def uu_ud(x):
    #     return inter_results[x][0][0] - inter_results[x][0][1]
    
    # F_difference_an = list(map(uu_ud, list(range(len(inter_results)))))

    # diff = list(map(lambda x,y: x-y, F_difference[0], F_difference[1]))
    # diff += list(map(lambda x,y: x-y, F_difference[-2], F_difference[-1]))
    diff = [np.array(F_difference[0])- np.array(F_difference[1])]
    diff += [np.array(F_difference[-2])- np.array(F_difference[-1])]
    # return F_difference + [F_difference_an], distances, [round(element, 3) for element in values] + ['analytical'], diff #second last entry is list of labels for later plotting
    return F_difference, distances,  [round(element, 3) for element in values], diff
    # return F_difference_an, distances, ['analytical'], diff

def interband(k, parameters, spin_orientation = [[0,1/2,0], [0,1/2,0]]):
    # parameters = [sites_x, sites_y, mu, cps, tri, gamma, jott, imp1, imp2]

    amplitudes, label = [], []

    step = 0.1
    
    for triplet in np.arange(0.3, parameters[4] + step , step):

        parameters[4] = triplet

        name = 'eigen data/eigen'
        for element in parameters:
            name += '_'+str(np.round(element,3))
        name += '.txt'

        if not os.path.exists(name): #check if eigenvalues were already calculated
            eigen = H_diag.diagonalize_hamiltonian(parameters, spin_orientation)
        
        else:
            eigen = load(name)

        coeffis = H_diag.operator_coefficients(eigen, 6)[0]

        res = o.interband_pairing(eigen, k[1:], coeffis)
       
        label.append(r'$V$ = '+str(round(triplet,3)))
     
        amplitudes.append(res)
    
    return amplitudes, label