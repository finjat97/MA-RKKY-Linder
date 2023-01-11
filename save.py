import numpy as np
import os

def make_folder(parameters):

    os.mkdir(os.getcwd()+'/data_'+ str(parameters[0])+'x'+str(parameters[1])+'_all')

    return True

def gap(data, parameters):

    with open('data_'+ str(parameters[0])+'x'+str(parameters[1])+'_all/gap_mu'+str(parameters[2])+'_'+ str(parameters[0])+'x'+str(parameters[1])+'_U'+ str(parameters[3])+'_V'+str(parameters[4])+'_gamma'+str(parameters[5])+'_J'+ str(parameters[6]) +'.txt', 'w') as file:
        for element in data:
            file.write(str(element)+'\n')
    
    return True

def ldos(data, parameters):

    with open('data_'+ str(parameters[0])+'x'+str(parameters[1])+'_all/ldos_mu'+str(parameters[2])+'_'+ str(parameters[0])+'x'+str(parameters[1])+'_U'+ str(parameters[3])+'_V'+str(parameters[4])+'_gamma'+str(parameters[5])+'_J'+ str(parameters[6]) +'_imp.txt', 'w') as file:
        for element in range(len(data)):
            file.write(str(data[element])+'\n')
    
    return True

def eigenvalues(data, parameters):

    with open('data_'+ str(parameters[0])+'x'+str(parameters[1])+'_all/eigenvalues_mu'+str(parameters[2])+'_'+ str(parameters[0])+'x'+str(parameters[1])+'_U'+ str(parameters[3])+'_V'+str(parameters[4])+'_gamma'+str(parameters[5])+'_J'+ str(parameters[6]) +'.txt', 'w') as file:
        for element in np.arange(data[0][0][0],data[0][0][-1]+0.2,0.01):
            file.write(str(element)+'\n')

    return True

def coeffies(data, parameters):

    with open('data_'+ str(parameters[0])+'x'+str(parameters[1])+'_all/coefficients_mu'+str(parameters[2])+'_'+ str(parameters[0])+'x'+str(parameters[1])+'_U'+ str(parameters[3])+'_V'+str(parameters[4])+'_gamma'+str(parameters[5])+'_J'+ str(parameters[6]) +'.txt', 'w') as file:
        for element in data:
            file.write(str(element))

    return True