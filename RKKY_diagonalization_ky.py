import numpy as np 
from scipy import sparse
import csv
from itertools import chain 

def diagonalize_hamiltonian(parameters, spin, positions = [], output=False):
    eigen = []

    sites_x, sites_y, mu, cps, tri, gamma, rkky, imp1, imp2 = parameters
    hopping = 1
    
    if len(positions) > 0:
        position = positions 
    else:
        position = [imp1, imp2]

    # calculate all allowed k-values 
    k_values = np.arange(-np.pi, np.pi ,2*np.pi/(sites_y))

    def matrix_elements(site, SOC_local, SOC_nn, off_diag_ham, attract, attract_tri_up, attract_tri_down):
        data, row_index, column_index = [] ,[], []
        # SOC_local = [-2*gamma*np.sin(k_y), -2*gamma*np.sin(k_y)]
        # SOC_nn = [-gamma, gamma]
        # off_diag_ham = [2*element *np.cos(k_y) for element in [-hopping, -hopping , hopping, hopping]]

         #local interactions
        ## chemical potential and singlet pairing
        data += [-mu, np.conj(attract[site//4]), -mu, np.conj(attract[site//4]), -attract[site//4], +mu , -attract[site//4], +mu]
        row_index += [site]*2 + [site+1]*2 +[site+2]*2 + [site+3]*2
        column_index += [site, site+3] + [site+1, site+2] + [site+1, site+2] + [site, site+3]
        # print(site, site+1, site+2, site+3, site+4)
        ## SOC
        data += SOC_local + [-element for element in SOC_local]
        row_index += [site, site+1, site+2, site+3]
        column_index += [site+1, site, site+3, site+2]
                    
        #RKKY-interaction (local)
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

            #to higher neighbor 
            # hopping and triplet pairing
            data += off_diag_ham + [attract_tri_up[site//4], np.conj(attract_tri_down[site//4]), -np.conj(attract_tri_up[site//4]), -(attract_tri_down[site//4])]# attract_tri_ham + [-np.conj(element) for element in attract_tri_ham]
            row_index += [site, site+1, site+2, site+3] * 2
            column_index += [site+4, site+4+1, site+4+2, site+4+3] + [site+4+2, site+4+3, site+4, site+4+1]
            #SOC
            data += SOC_nn + [-element for element in SOC_nn] #SOC for nearest neighbors, second half negative since c^dag c = -c c^dag
            row_index += [site, site+1, site+2, site+3]
            column_index += [site+4+1, site+4, site+4+3, site+4+2]

            # if len(data) != len(row_index) or len(data) != len(column_index):
                # print(len(data), len(row_index), len(column_index))
            
        if (site - 4) >=0:
            #to lower neighbor
            # hopping and triplet pairing
            data += off_diag_ham + [-(attract_tri_up[site//4]), -np.conj(attract_tri_down[site//4]),np.conj(attract_tri_up[site//4]), attract_tri_down[site//4]]# [np.conj(element) for element in attract_tri_ham] +[-element for element in attract_tri_ham]
            row_index += [site, site+1, site+2, site+3] * 2
            column_index += [site-4, site-4+1, site-4+2, site-4+3]  + [site-4+2, site-4+3, site-4, site-4+1]
            #SOC
            data += [-entry for entry in SOC_nn + [-element for element in SOC_nn]] #opposite direction as before (line 64), which means opposite sign for SOC
            row_index += [site, site+1, site+2, site+3]
            column_index += [site-4+1, site-4, site-4+3, site-4+2]
    
        return data, row_index, column_index

    def k_matrices(k_y):
        SOC_local = [-2*gamma*np.sin(k_y), -2*gamma*np.sin(k_y)]
        SOC_nn = [-gamma, gamma]
        off_diag_ham = [2*element *np.cos(k_y) for element in [-hopping, -hopping , hopping, hopping]]
        attract = [cps]*sites_x*sites_y
        attract_tri_up = [tri*np.cos(k_y)]*sites_x*sites_y
        attract_tri_down = [tri*np.cos(k_y)]*sites_x*sites_y

        # find entries for all sites of the system
        site_loop = []
        for site in range(0, sites_x*4, 4):
            site_loop += (matrix_elements(site, SOC_local, SOC_nn, off_diag_ham, attract, attract_tri_up, attract_tri_down))

        # extract information from site_loop and flatten resulting lists of arrays
        data = list(chain(*site_loop[0::3])) #np.concatenate(list(map(d, list(range(sites_x))))).ravel()
        row_index = list(chain(*site_loop[1::3])) #np.concatenate(list(map(r, list(range(sites_x))))).ravel()
        column_index = list(chain(*site_loop[2::3])) #np.concatenate(list(map(c, list(range(sites_x))))).ravel()
        # create matrix from obtained information
        matrix = sparse.csr_matrix((data, (row_index, column_index))).todense()

        #return eigenvectors and eigenvalues for the given k
        res = np.linalg.eigh(matrix)
        eigvec = np.array(res[1])
        eigval = res[0].reshape(4*sites_x,1)
        res2 = np.concatenate((eigval, eigvec), axis=1)
        #return np.linalg.eigh(matrix)
        return res2

    #calculate eigenvectors and eigenvalues for all allowed k values
    #print('k_values ', k_values.shape)
    eigen = (list(map(k_matrices, k_values)))
    eigen2 = np.array(eigen) #shape = (N_y, 4*N_x, 4*n_x +1), first column on third axis is eigenvalues, N_y is number of k-values

    return eigen2

def operator_coefficients(diagonal, k):
    ## define u and v to get hamiltonian on form of Fermi-gas (see notes "realspace.pdf")
    # u_up, u_down, v_up, v_down = [], [], [], [] #lists of lists

    # u_up += [np.conj(element) for element in diagonal[k][1][range(0, len(diagonal[0][0]),4)]]#appending N_x coefficients for one site, one k and all energy eigenvalues
    # u_down += [np.conj(element) for element in diagonal[k][1][np.add(range(0, len(diagonal[0][0]),4),1)]]
    # v_up += [np.conj(element) for element in diagonal[k][1][np.add(range(0, len(diagonal[0][0]),4),2)]]
    # v_down += [np.conj(element) for element in diagonal[k][1][np.add(range(0, len(diagonal[0][0]),4),3)]]
    u_up_indices = np.arange(0+1, diagonal.shape[2], 4)
    u_down_indices = np.arange(1+1, diagonal.shape[2], 4)
    v_up_indices = np.arange(2+1, diagonal.shape[2], 4)
    v_down_indices = np.arange(3+1, diagonal.shape[2], 4)

    u_up = diagonal[:, diagonal.shape[1]//2:, u_up_indices] # shape = (sites_y, positive eigenenergies , sites_x)
    u_down = diagonal[:, diagonal.shape[1]//2: , u_down_indices]
    v_up = diagonal[:, diagonal.shape[1]//2: , v_up_indices]
    v_down = diagonal[:, diagonal.shape[1]//2: , v_down_indices]

    return u_up, u_down, v_up, v_down

# diag = diagonalize_hamiltonian([21, 15, 0.5, 0.5, 0.1, 0.15, 2, 3, 7],  [[0,1/2,0],[1/2,0,0]])
# operator_coefficients(diag, 5)
