import numpy as np 
import scipy as sp
import csv

def diagonalize_hamiltonian(parameters, spin, positions = [], output=False):
    eigen = []

    sites_x, sites_y, mu, cps, tri, gamma, rkky, imp1, imp2 = parameters
    hopping = 1

    attract = [cps]*sites_x*sites_y
    attract_tri_up = [tri]*sites_x*sites_y
    attract_tri_down = [tri]*sites_x*sites_y
    if len(positions) > 0:
        position = positions 
    else:
        position = [imp1, imp2]

    # calculate all allowed k-values 
    k_values = np.arange(-np.pi, np.pi ,2*np.pi/(sites_y))

    # basis: nambu; go through all sites in x-direction and do that for all k-values
    for k_y in k_values:
        data, row_index, column_index = [] ,[], []
        SOC_ham = [-gamma*np.cos(k_y), gamma*np.sin(k_y), gamma*np.sin(k_y), gamma*np.cos(k_y), gamma*np.cos(k_y), -gamma*np.sin(k_y), -gamma*np.sin(k_y), -gamma*np.cos(k_y)]
        off_diag_ham = [element *np.cos(k_y) for element in [-hopping, -hopping , hopping, hopping]]

        for site in range(0, sites_x*4, 4):
                
            #local interactions
            data += [-mu, np.conj(attract[site//4]), -mu, -np.conj(attract[site//4]), attract[site//4], +mu , - attract[site//4], +mu]
            row_index += [site]*2 + [site+1]*2 +[site+2]*2 + [site+3]*2
            column_index += [site, site+3] + [site+1, site+2] + [site+1, site+2] + [site, site+3]
                        
            #RKKY
            if site//4 in position: 
                # all c^dag c terms
                current_spin = spin[position.index(site//4)%2]
                data += [rkky*current_spin[2], (rkky*current_spin[0]-1j*rkky*current_spin[1]), (rkky*current_spin[0]+1j*rkky*current_spin[1]), -rkky*current_spin[2]]
                row_index += [site, site, site+1, site+1]
                column_index += [site, site +1, site, site+1]
                # all c c^dag terms
                data += [-rkky*current_spin[2], (rkky*current_spin[0]+1j*rkky*current_spin[1]), (rkky*current_spin[0]-1j*rkky*current_spin[1]), rkky*current_spin[2]]
                row_index += [site+2, site+2, site+3, site+3]
                column_index += [site+2, site +3, site+2, site+3]
                                

            #nearest neighbor hopping only, hard wall boundary conditions, 1D x-direction
            if site//4 != sites_x-1:
                #hopping to higher neighbor
                data += off_diag_ham + [attract_tri_up[site//4], attract_tri_down[site//4],-np.conj(attract_tri_up[site//4]), -np.conj(attract_tri_down[site//4])]
                row_index += [site, site+1, site+2, site+3] * 2
                column_index += [site+4, site+4+1, site+4+2, site+4+3] + [site+4+2, site+4+3, site+4, site+4+1]
                data += SOC_ham
                row_index += [site]*2 + [site+1]*2 + [site+2] *2 + [site+3]*2
                column_index += [site+4, site+4+1] + [site+4, site+4+1] + [site+4+2,site+4+3] + [site+4+2, site+4+3]

                if len(data) != len(row_index) or len(data) != len(column_index):
                    print(len(data), len(row_index), len(column_index))
                
            if (site - 4) >=0:
                #hopping to lower neighbor
                data += off_diag_ham + [np.conj(attract_tri_up[site//4]), np.conj(attract_tri_down[site//4]),-attract_tri_up[site//4], -attract_tri_down[site//4]]
                row_index += [site, site+1, site+2, site+3] * 2
                column_index += [site-4, site-4+1, site-4+2, site-4+3]  + [site-4+2, site-4+3, site-4, site-4+1]
                data += SOC_ham
                row_index += [site]*2 + [site+1]*2 + [site+2] *2 + [site+3]*2
                column_index += [site-4, site-4+1] + [site-4, site-4+1] + [site-4+2,site-4+3] + [site-4+2, site-4+3]
                                
            else:
                continue
        
        # print(sp.sparse.csr_matrix((data, (row_index, column_index))).todense().shape)
        eigen.append(np.linalg.eigh(sp.sparse.csr_matrix((data, (row_index, column_index))).todense()))  #look for intel version 

    if output:
        print('__hamiltonian__\n diagonalized ')

    return eigen

def operator_coefficients(diagonal, k):
    ## define u and v to get hamiltonian on form of Fermi-gas (see notes "realspace.pdf")
    u_up, u_down, v_up, v_down = [], [], [], [] #lists of lists

    u_up += [np.conj(element) for element in diagonal[k][1][range(0, len(diagonal[0][0]),4)]]#appending N_x coefficients for one site, one k and all energy eigenvalues
    u_down += [np.conj(element) for element in diagonal[k][1][np.add(range(0, len(diagonal[0][0]),4),1)]]
    v_up += [np.conj(element) for element in diagonal[k][1][np.add(range(0, len(diagonal[0][0]),4),2)]]
    v_down += [np.conj(element) for element in diagonal[k][1][np.add(range(0, len(diagonal[0][0]),4),3)]]


    return u_up, u_down, v_up, v_down
