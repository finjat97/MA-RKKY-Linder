import numpy as np 
from scipy import sparse
from itertools import chain 

def diagonalize_hamiltonian(parameters, spin, positions = [], output=False):
    
    sites_x, sites_y, mu, cps, tri, gamma, rkky, imp1, imp2 = parameters
    ds = cps
    dtu = tri 
    dtd = tri
    hopping = 1
    
    if len(positions) > 0:
        position = positions 
    else:
        position = [imp1, imp2]
        
    k_values = np.arange(-np.pi, np.pi ,2*np.pi/(sites_y))

    def k_matrices(k_y):

        diag = np.array([-mu, -2*gamma*np.sin(k_y), 0, np.conj(ds), -2*gamma*np.sin(k_y), -mu, - np.conj(ds), 0, 0, -ds, mu, 2*gamma*np.sin(k_y), ds, 0, 2*gamma*np.sin(k_y), mu]).reshape(4,4)
        # off_diag = np.array( [-2*hopping*np.cos(k_y), -gamma, 2*np.cos(k_y)*dtu,0,gamma, -2*hopping*np.cos(k_y), 0, 2*np.cos(k_y)*np.conj(dtd), -2*np.cos(k_y)*np.conj(dtu), 0, 2*hopping*np.cos(k_y), gamma, 0, -2*np.cos(k_y)*dtd, - gamma, 2*hopping*np.cos(k_y)] ).reshape(4,4)
        off_diag = np.array( [-2*hopping*np.cos(k_y), -gamma, 2*dtu,0,gamma, -2*hopping*np.cos(k_y), 0, 2*np.conj(dtd), -2*np.conj(dtu), 0, 2*hopping*np.cos(k_y), gamma, 0, -2*dtd, - gamma, 2*hopping*np.cos(k_y)] ).reshape(4,4)

        matrix = np.zeros((sites_x*4, sites_x*4), dtype='complex128')

        for i in np.arange(0,sites_x*4,4):
            matrix[i:i+4, i:i+4] = diag
            if i < (sites_x-1)*4 :
                matrix[i:i+4, i+4:i+8] = off_diag
                matrix[ i+4:i+8,i:i+4] = np.conj(off_diag.T)
            if i//4 in position:
                current_spin = spin[position.index(i//4)%2]
                data = np.asarray([rkky*current_spin[2], (rkky*current_spin[0]-1j*rkky*current_spin[1]), (rkky*current_spin[0]+1j*rkky*current_spin[1]), -rkky*current_spin[2]]).reshape(2,2)
                
                matrix[i:i+2, i:i+2] = data
                matrix[i+2:i+4, i+2:i+4] = -data
        
        k_eig = np.linalg.eigh(matrix)
        eigvec = np.array(k_eig[1])
        eigval = k_eig[0].reshape(4*sites_x,1)
        res = np.concatenate((eigval, eigvec), axis=1)

        return res
    
    #calculate eigenvectors and eigenvalues for all allowed k values
    eigen = np.array(list(map(k_matrices, k_values)))

    return eigen

# out = diagonalize_hamiltonian([81,81,0.5,0.1, 0.01, 0.2, 2, 2, 12], [[0,1/2,0], [0,1/2,0]])

def operator_coefficients(diagonal, k):
    
    u_up_indices = np.arange(0+1, diagonal.shape[2], 4)
    u_down_indices = np.arange(1+1, diagonal.shape[2], 4)
    v_up_indices = np.arange(2+1, diagonal.shape[2], 4)
    v_down_indices = np.arange(3+1, diagonal.shape[2], 4)

    u_up = diagonal[:, diagonal.shape[1]//2:, u_up_indices] # shape = (sites_y, positive eigenenergies , sites_x)
    u_down = diagonal[:, diagonal.shape[1]//2: , u_down_indices]
    v_up = diagonal[:, diagonal.shape[1]//2: , v_up_indices]
    v_down = diagonal[:, diagonal.shape[1]//2: , v_down_indices]

    return u_up, u_down, v_up, v_down