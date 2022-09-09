import numpy as np 
import scipy as sp
import time as timer
import matplotlib.pyplot as plt

def construct_hamiltonian(sites_x, sites_y, hopping, mu, gamma_uu, gamma_dd, gamma_ud, gamma_du, attract, spin, position, output=False):
    
    dim = (sites_x * sites_y)*4
    data, row_index, column_index = [] ,[], []
    atousa = 0
    ### blocks for hamiltonian
    off_diag_ham = [hopping, hopping, -hopping, -hopping] + [-2*atousa, 2* atousa , np.conj(2*atousa),-np.conj(2*atousa)]
    local_diag_ham = [-mu + gamma_uu, gamma_ud, attract, gamma_du, -mu + gamma_dd, -attract, -np.conj(attract), mu-gamma_uu, -gamma_du, np.conj(attract),  -gamma_ud, mu -gamma_dd]
    spin_spin_ham = [spin[2], spin[0]-1j*spin[1], (spin[0]+1j*spin[1]), spin[2]]

    #### build hamiltonian ####
    # basis: nambu
    for row in range(0,dim,4):
        for column in range(0,dim,4):
            
            #local interactions
            if row == column: 
                data += local_diag_ham
                row_index += [row, row, row, row+1, row+1, row+1, row+2, row+2, row+2, row+3, row+3, row+3 ]
                column_index += [column, column+1, column+3, column, column+1, column+2, column+1, column+2, column+3, column, column+2, column+3]

                #RKKY
                if row in [ (element-1)*4 for element in position]: 
                    # all c^dag c terms
                    data += spin_spin_ham
                    row_index += [row, row, row+1, row+1]
                    column_index += [column, column +1, column, column+1]
                    # all c c^dag terms
                    data += [-element for element in spin_spin_ham]
                    row_index += [row+2, row+2, row+3, row+3]
                    column_index += [column+2, column +3, column+2, column+3]

            #nearest neighbor hopping only, hard wall boundary conditions, 1D x-direction
            elif abs(row-column) == 4 and row < column:
                # print(int(row/4+1),int(column/4+1))
                if int(row/4+1)%sites_x != 0:
                    # direction one
                    data += off_diag_ham
                    row_index += [row, row+1, row+2, row+3] *2 
                    column_index += [column, column+1, column+2, column+3] + [column+3, column+2, column+1, column]
                    # print(1, int(row/4+1), int(column/4+1))
                    # opposite direction
                    data += off_diag_ham
                    row_index += [column, column+1, column+2, column+3] * 2 
                    column_index += [row, row+1, row+2, row+3] + [row+3, row+2, row+1, row]
                    # print(2, int(column/4+1), int(row/4+1))

            elif abs(row-column) == sites_x*4:          
                data += off_diag_ham
                row_index += [row, row+1, row+2, row+3] *2 
                column_index += [column, column+1, column+2, column+3] + [column+3, column+2, column+1, column]
                # print(4, int(row/4+1), int(column/4+1))
            
            else:
                continue
    if output:
        print('__hamiltonian__\n constructed')

    return sp.sparse.csr_matrix((data, (row_index, column_index)))

def delta_function(x, width): #dirac delta function approximated as rectangle

    if x <= width and -width <= x:
        value = 1/(2*width)
    else: value = 0

    return value

def density_of_states(eigen, coefficients, site, output=False):
    
    density = [] #list of densities for each site
    u_up, u_down, v_up, v_down = coefficients[0], coefficients[1], coefficients[2], coefficients[3]    
    
    for energy in np.arange(eigen[0][0],eigen[0][-1]+0.2,0.001):
        summand_1, summand_2 = 0, 0
        for eigenvalue in range(int(len(eigen[0])/2)):
            summand_1 += (abs(v_down[site][2*eigenvalue])**2 + abs(v_up[site][2*eigenvalue])**2)*delta_function(energy+eigen[0][2*eigenvalue],0.3)
            summand_2 += (abs(u_down[site][2*eigenvalue])**2 + abs(u_up[site][2*eigenvalue])**2)*delta_function(energy-eigen[0][2*eigenvalue],0.3)
        density += [summand_1+summand_2]
    
    if output: 
        #print('__density of states__\n', [np.round(element,4) for element in density])
        print(round(sum(element for element in density),4))
    
    return density

def fermi_distribution(energy):
    constant = 1 # k_B*T

    return 1/ (np.exp(energy/constant)+1)

def operator_coefficients(diagonal):
    ## define u and v to get hamiltonian on form of Fermi-gas (see notes "realspace-pdf")
    u_up, u_down, v_up, v_down = [], [], [], []
    matrix = np.matrix.getH(diagonal[1])

    for row in range(0,len(diagonal[0]),4): 
        u_up.append([element for element in matrix[row]])
        u_down.append([element for element in matrix[row+1]])
        v_up.append([element for element in matrix[row+2]])
        v_down.append([element for element in matrix[row+3]])

    return u_up, u_down, v_up, v_down

def selfconsistency(eigen, potential,coefficients, output=False):

    u_up, u_down, v_up, v_down = coefficients[0], coefficients[1], coefficients[2], coefficients[3]

    #summation over all eigenenergies, gap is list with gaps for each site
    gap = [np.round(potential*sum(u_down[site][value]*np.conj(v_up[site][value])*fermi_distribution(eigen[0][value])
            +u_up[site][value]*np.conj(v_down[site][value])*(1-fermi_distribution(eigen[0][value])) for value in range(len(eigen[0]))),5) for site in range(int(len(eigen[0])/4))]

    gap_real = [float(element) for element in gap]

    if output:
        print('__gap__\n', gap_real)

    return gap_real

def main():
    ### system parameters ###
    sites_x, sites_y = 10,10

    ## all paramters expressed in terms of hopping
    mu = 0.7 #chemical potential
    hopping = 1 #hopping energy
    attract = 2 #attractive potential aka. Cooper pairing strength
    gamma_uu, gamma_dd, gamma_ud, gamma_du = 1,-1,0,0#3,-3 #spin-orbit-coupling strength, spin-dependent
    jott = 3 #local-itinerant spin interaction strength
    spin = [jott*0.5, jott*0.5, jott*0.5] #spin components of impurity spin
    spin_positions = [1,4]

    hamiltonian = construct_hamiltonian(sites_x, sites_y, hopping, mu, gamma_uu, gamma_dd, gamma_ud, gamma_du, attract, spin, spin_positions)    
    eigen = np.linalg.eigh(np.asarray(hamiltonian.todense())) #first entry = eigenvalues in ascending order, second entry = eigenvectors to the eigenvalues
    coeffis = operator_coefficients(eigen)

    print('__eigenvalues__\n', [np.round(element, 4) for element in eigen[0]]) # I can't see a symmetry in the eigenvalues -> what is wrong? + they're complex, when I include RKKY
    
    site = 7
    densities = density_of_states(eigen, coeffis, site, output=False)
    
    plt.plot(np.arange(eigen[0][0],eigen[0][-1]+0.2,0.001), densities)
    plt.xlabel('energy in E/t')
    plt.ylabel('LDOS')
    plt.title('Local density of states at site '+str(site)+' of '+ str(sites_x)+'x'+str(sites_y))
    plt.grid()
    # plt.savefig('ldos_'+str(sites_x)+'x'+str(sites_y)+'_'+str(site)+'.png')
    plt.show()

    # gap_equation = selfconsistency(eigen, attract, coeffis, output=True) #depends on system parameters, but not on the actual state

    # plt.plot(gap_equation)
    # plt.grid()
    # plt.show()

    return True 

if __name__ == '__main__':
    start = timer.time()

    if main(): 
        print('duration ', round(timer.time()-start, 4))
