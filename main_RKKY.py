import numpy as np 
import time as timer
import argparse
import plot
import routines

def main(sites_x, sites_y, mu, cps, tri, jott, gamma, imp1, imp2):
    ### system parameters ###
    parameters = [sites_x, sites_y, mu, cps, tri, gamma, jott, imp1, imp2]

    ## all paramters expressed in terms of hopping which is set to 1
    spin_1 = [0.5, 0, 0] #spin components of impurity spin
    spin_2 = [0, 0.5, 0]

    spin_orientation = [spin_1, spin_2]
    spin_positions = [imp1,imp2]

    # attract = [cps]*sites_x*sites_y
    # attract_tri = [tri]*sites_x *sites_y
    
    kvalues = list(np.arange(-np.pi, np.pi ,2*np.pi/(sites_y))) # N_y k-values
    positive_kvalues = [kvalues.index(element) for element in kvalues if round(element,6) > 0] #indices of positive k-values
    zero_kvalues = [kvalues.index(element) for element in kvalues if round(element,6) == 0]#indicies of zero k-values
    
    #### what is the program supposed to do?
    iter_bool = False
    spin_opti = False
    triplet_study = False
    SOC_study = False
    RKKY_study = True
    gap_3D = False

    #### let's get to work
    if spin_opti:
        positions =[[5, 10], [5, 25], [5,35]] # [[1,2],[1,3]] #
        spin_configs, free_energies = [], []
        for pos in positions: # go through the possible positions for the impurity spins
            configs, energies = routines.spin_loop(parameters, kvalues, pos, iteration=iter_bool) # calculate free energy for different orientations of the impurity spins
            spin_configs = configs # save the orientation of the impurity spins for which the free energy was calculated
            free_energies.append(energies)
        labels = ['i = '+str(element) for element in positions]
        plot.spinphase(free_energies, spin_configs, parameters, labels, output=False)

    if triplet_study:
        # calculate density, singlet gap and triplet gap for different values of the triplet gap (potential), can be self-consistent
        study_results = routines.gap_study(parameters, [kvalues, positive_kvalues, zero_kvalues], spin_orientation, spin_positions) #[gap_studies,  energies, density_study, parameters, labels, site]
        plot.density(study_results[1], study_results[2], study_results[5], study_results[3], study_results[4], 'tri', output=True)
        # plot.gap(routines.gap_eff(sites_x, study_results[0], tri, study_results[6])[1], parameters, study_results[4], 'tri')
        # plot.spinphase(study_results[8], study_results[7], parameters, study_results[4])
        print(routines.zero_LDOS_gap(parameters, study_results[2], study_results[1], study_results[6]))

    if SOC_study:
        # for different gamma values: density, singlet gap and triplet gap
        study_results = routines.soc_study(parameters, [kvalues, positive_kvalues, zero_kvalues], spin_orientation, spin_positions)
        plot.density(study_results[1], study_results[2], study_results[5], study_results[3], study_results[4], 'soc')
        # plot.gap(routines.gap_eff(sites_x, study_results[0], gamma, study_results[6])[1], parameters, study_results[4], 'soc')
        if len(study_results[8]) > 3:
            plot.spinphase(study_results[8], study_results[7], parameters, study_results[4])

    if RKKY_study:
        # for different jott values: density, singlet gap and triplet gap
        study_results = routines.rkky_study(parameters, [kvalues, positive_kvalues, zero_kvalues], spin_orientation, spin_positions)
        plot.density(study_results[1], study_results[2], study_results[5], study_results[3], study_results[4]+[element+' tri' for element in study_results[4]], 'rkky', output=True)
        # plot.gap(routines.gap_eff(sites_x, study_results[0], jott, study_results[6])[1], parameters, study_results[4], 'rkky', abs=False)

    if gap_3D:
        gap = []
        for rkky in range(0,10,2):
            parameters[6] = rkky
            study_results = routines.gap_study(parameters, [kvalues, positive_kvalues, zero_kvalues], spin_orientation, spin_positions) #[gap_studies,  energies, density_study, parameters, labels, site]
            plot.density(study_results[1], study_results[2], study_results[5], study_results[3], study_results[4], 'tri', output=False)
            gap.append(routines.zero_LDOS_gap(parameters, study_results[2], study_results[1], study_results[6]))
        
        plot.gap_3D(range(0,10,2), np.arange(0.5, parameters[4] + study_results[6] , study_results[6]), gap, parameters)

    return True 

if __name__ == '__main__':
    start = timer.time()

    parser = argparse.ArgumentParser(description='Local density of states and gap equation for 2d superconductor (non-centro, incl. RKKY')

    parser.add_argument('sites_x', metavar='N_x', type=int, help='number of sites in x direcion') 
    parser.add_argument('sites_y', metavar='N_y', type=int, help='number of sites in y direcion') 
    parser.add_argument('chemical', metavar='mu', type=float, help='chemical potential') 
    parser.add_argument('attract', metavar='delta', type=float, help='electron pair interaction strength - singlet') 
    parser.add_argument('tri', metavar='delta_tri', type=float, help='electron pair interaction strength - triplet') 
    parser.add_argument('soc', metavar='gamma', type=float, help='spin-orbit-coupling strength') 
    parser.add_argument('rkky', metavar='jott', type=float, help='RKKY interaction strength')
    parser.add_argument('impure1_x', metavar='impurity1_x', type=int, help='position of first impurity x-coordinate')
    parser.add_argument('impure2_x', metavar='impurity2_x', type=int, help='position of second impurity x-coordinate')

    args = parser.parse_args()

    if main(args.sites_x, args.sites_y, args.chemical, args.attract, args.tri, args.rkky, args.soc, args.impure1_x, args.impure2_x): 
        print('total duration ', round(timer.time()-start, 4))
