import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from pylab import figure, cm

# parameters = [sites_x, sites_y, mu, cps, tri, gamma, jott, imp1, imp2]

def density(eigen, y_data, site, parameters, labels, call, output=False):

    if type(y_data[0]) is list:
        for element in range(len(y_data)):
            plt.plot(np.arange(eigen[element][0][0][0],eigen[element][0][0][-1]+0.01,0.01), y_data[element], label=labels[element])
        

    else: 
        plt.plot(np.arange(eigen[0][0][0],eigen[0][0][-1]+0.2,0.01), y_data, label=r'$\gamma$ ='+str(round(parameters[5],2)) + ', J= '+str(round(parameters[6],2))+' and i='+str(round(parameters[7],2))+','+str(round(parameters[8],2)))
    
    
    plt.title('LDOS for site '+str(site)+' in system '+str(round(parameters[0],2))+'x'+str(round(parameters[1],2))+'\n'+r' $U=$'+str(round(parameters[3],2))+r', $V=$'+str(round(parameters[4],2))+r', $\gamma=$'+str(round(parameters[5],2))+r', $J=$'+str(round(parameters[6],2))+r', $i=$'+str(round(parameters[7],2))+', '+str(round(parameters[8],2)))

    plt.xlabel('energy in 1/t')
    plt.ylabel('LDOS')
    plt.legend()
    plt.grid()

   
    name = 'ldos_'+str(call)
    for element in parameters:
        name += '_'+str(np.round(element,2))
    name += '.png'
    plt.savefig(name)
    
    if output: plt.show()
    else: plt.clf()

    if type(y_data[0]) is list:
        for element in range(len(y_data)):
            derivative = [(y_data[element][entry]-y_data[element][entry-1])/0.01 for entry in np.arange(1,len(y_data[element]))]
            derivative = [(derivative[entry]-derivative[entry-1])/0.01 for entry in np.arange(1,len(derivative))]
            plt.plot(np.arange(eigen[element][0][0][0]+0.01,eigen[element][0][0][-1],0.01), derivative, label=str(element))
    plt.legend()
    plt.show()
    
    return True

def gap(gap, parameters, labels, call, abs= True, output=False):

    if type(gap[0]) is list:
        for element in range(len(gap)):
            if abs: plt.plot(range(1,len(gap[element])+1), np.square(np.absolute(gap[element])), label=labels[element])
            else: plt.plot(range(1,len(gap[element])+1), gap[element], label=labels[element])
        plt.title('relative gap for each site in system '+str(round(parameters[0],2))+'x'+str(round(parameters[1],2))+r' with $U=$'+str(round(parameters[3],2))+r', $\gamma=$'+str(round(parameters[5],2))+r', $J=$'+str(round(parameters[6],2)))

    else:
        plt.plot(range(1,len(gap)+1), [abs(element)**2 for element in gap], label=r'$\gamma$ ='+str(round(parameters[5],2)) + ', J= '+str(round(parameters[6],2))+' and i='+str(round(parameters[7],2))+','+str(round(parameters[8],2)))
        plt.title('relative gap for each site in system '+str(round(parameters[0],2))+'x'+str(round(parameters[1],2))+r' with $U=$'+str(round(parameters[3],2))+r' with $V=$'+str(round(parameters[4],2)))
    
    plt.xlabel('site i')
    plt.ylabel('gap')
    plt.legend()
    plt.grid()
    
    name = 'gap_'+str(call)
    for element in parameters:
        name += '_'+str(np.round(element,2))
    name += '.png'
    plt.savefig(name)

    if output: plt.show()

    plt.clf()

    return True

def spinphase(free_energy, spin_config, parameters, legend, output=False):

    if type(free_energy[0]) is list:
        for element in range(len(free_energy)):
            plt.plot(range(len(spin_config)), free_energy[element], label=legend[element])

    else: plt.plot(range(len(spin_config)), free_energy)

    labels =[]
    for entry in spin_config:
        labels += ['('+entry[0]+entry[1]+')']
    # for element in spin_config:
    #     component = element.index()

    plt.xlabel('spin configuration')
    plt.ylabel('free energy in 1/t')
    plt.xticks(range(len(spin_config)), labels, rotation=45)
    plt.title('free energy for different spin-orientations of impurity spins \n for '+str(parameters[0])+'x'+str(parameters[1])+r'sites with $U=$'+str(parameters[3])+r' and $V=$'+str(parameters[4])+', J='+str(parameters[6])+r', $\gamma=$'+str(round(parameters[5],2)))
    plt.tight_layout()
    plt.legend()
    plt.grid()
    
    name = 'spinstructure'
    for element in parameters:
        name += '_'+str(np.round(element,2))
    name += '.png'
    
    plt.savefig(name)
    if output: plt.show()
    plt.clf()

    return True

def gap_3D(x_values, y_values, z_values, param):

    # fig = plt.figure()
    # ax = plt.axes(projection = '3d')

    # ax.plot3D(x_values, y_values, z_values)
    # ax.set_title('gap for different values of J and V for U = '+str(param[3])+r' $\gamma$ = '+str(param[5])+' at impurity site')

    x1, x2 = np.meshgrid(x_values, y_values)
    plt.imshow(z_values, extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]], cmap=cm.jet, origin = 'lower')
    plt.colorbar()
    plt.xticks(range(len(x_values)), x_values)
    plt.yticks(range(len(y_values)), y_values)
    plt.xlabel('RKKY')
    plt.ylabel('V')
    
    plt.show()

    return True

def analytical_vs_numerical(data_a, data_n, parameters_a, parameters_n, spin_config):

    labels =[]
    for entry in spin_config:
        labels += ['('+entry[0]+entry[1]+')']

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('spin orientation')
    ax1.set_ylabel('F in t')
    ax1.plot(data_n, color='black', label = 'numerical')
    plt.xticks(range(len(spin_config)), labels, rotation=45)

    ax2 = ax1.twinx()
    ax2.set_ylabel('F in t', color='green')
    ax2.plot(data_a, color = 'green', label = ' analytical')
    # ax1.xticks(range(len(spin_config)), labels, rotation=45)
    # ax2.tick_params()

    # plt.plot(data_a, color='green', label='analytical')
    # plt.plot(data_n, color = 'black', label='numerical')
    plt.legend()
    plt.title('numerical and analytical spin structure')
    

    plt.savefig('spinstructure_comparison.png')
    plt.clf()
    # plt.show()

    return True

def test(x_values, y_values, details):

    plt.plot(x_values, y_values)
    plt.grid()
    plt.title(str(details))
    plt.show()

    return True

def distance(x_values, y_values, parameters, labels, call, output=False):

    if len(y_values) > 1:
        for index in range(len(y_values)):
            plt.plot(x_values, y_values[index], label=labels[index])
    else:
        plt.plot(x_values, y_values)

    plt.xlabel(r'distance in $a$')
    plt.ylabel(r'$F_{upup}-F_{downdown}$')
    plt.title('free energy difference for different impurity separations')
    plt.legend()
    plt.grid()
    
    name = 'dF_'+str(call)
    for element in parameters:
        name += '_'+str(np.round(element,2))
    name += '.png'
    plt.savefig(name)
    
    if output: plt.show()
    else: plt.clf()

    return True