import read
#import subduction
#import boot

################################################################################
# parameters for reading #######################################################
sta_cnfg = 714
end_cnfg = 1214
del_cnfg = 2

T = 48        # number of timeslices
#p = 2         # momenta

verbose = 0

directory = '/hiskp2/knippsch/Rho_Feb2016/'

missing_configs = [1282]

# parameters for subduction #######################################################
# Operators entering the Gevp. Last entry must contain the name in LaTeX 
# compatible notation for plot labels
gamma_i =   [1, 2, 3, '\gamma_i']
gamma_0i =  [10, 11, 12, '\gamma_0\gamma_i']
gamma_50i = [15, 14, 13, '\gamma_5\gamma_0\gamma_i']

gamma = [gamma_i, gamma_0i, gamma_50i]
gamma_for_filenames = {'\gamma_i' : 'gi', '\gamma_0\gamma_i' : 'g0gi', \
                       '\gamma_5\gamma_0\gamma_i' : 'g5g0gi'}

# parameters for bootstrap #######################################################
bootstrap_original_data = True
nb_boot = 2000

################################################################################

for p in [0]:
  read.read_ensembles(sta_cnfg, end_cnfg, del_cnfg, p, T, directory, missing_configs, verbose)
#  subduction.subduce_ensembles(p, gamma, gamma_for_filenames, verbose)
#  boot.bootstrap_ensembles(p, nb_boot, bootstrap_original_data)
