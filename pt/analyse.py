#!/hiskp2/werner/libraries/Python-2.7.12/python

import numpy as np

import read
import wick

################################################################################
# parameters ###################################################################

sta_cnfg = 714
end_cnfg = 730
del_cnfg = 2

T = 48        # number of timeslices
p_max = 4
p = range(2)         # momentum
p_cm_max = np.asarray((4,5,6,7,4), dtype=int)

# gamma structure wich shall be averaged. Last entry of gamma must contain all
# names in LaTeX compatible notation for plot labels
gamma_i =   [1, 2, 3, \
             ['\gamma_1', '\gamma_2', '\gamma_3', '\gamma_i']]
gamma_0i =  [10, 11, 12, \
             ['\gamma_0\gamma_1', '\gamma_0\gamma_2', '\gamma_0\gamma_3', \
              '\gamma_0\gamma_i']]
gamma_50i = [13, 14, 15, \
             ['\gamma_5\gamma_0\gamma_1', '\gamma_5\gamma_0\gamma_2', \
              '\gamma_5\gamma_0\gamma_3', '\gamma_5\gamma_0\gamma_i']]
gamma_5 = [5, ['\gamma_5']]

gammas = [gamma_i, gamma_0i, gamma_50i]

#diagrams = ['C20', 'C3+', 'C4+D', 'C4+B']
diagrams = ['C20']

directories = ['/hiskp2/knippsch/Rho_Jun2016/']
#directories = ['/hiskp2/knippsch/Rho_Jun2016/', \
#               '/hiskp2/knippsch/Rho_Jun2016/', \
#               '/hiskp2/knippsch/Rho_A40.24/', \
#               '/hiskp2/knippsch/Rho_Jun2016/']

verbose = 0

missing_configs = [1282]

###############################################################################

def main():

  for p_cm in p:
    for diagram, directory in zip(diagrams, directories):
      read.ensembles(sta_cnfg, end_cnfg, del_cnfg, diagram, p_cm, p_cm_max, \
                     p_max, gammas, T, directory, missing_configs, verbose)
    wick.rho_2pt(p_cm, 'C20', verbose)

#    for diagram, directory in zip(diagrams, directories):
#      subduce.ensembles()

if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    pass


