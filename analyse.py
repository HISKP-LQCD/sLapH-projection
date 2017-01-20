#!/hiskp2/werner/libraries/Python-2.7.12/python
import argparse
import ConfigParser

import numpy as np

import read
import wick

################################################################################
# Parameters ###################################################################

################################################################################
# Argument parsing ############################################################# 

parser = argparse.ArgumentParser()
# parse the name of the infile and load its contents into the parser
parser.add_argument("infile", help="name of input file")
# verbosity is also parsed
parser.add_argument("-v", "--verbose", action="store_true", \
                                               help="increase output verbosity")
args = parser.parse_args()
verbose = args.verbose

################################################################################
# Reading infile ###############################################################

config = ConfigParser.RawConfigParser()

if(config.read(args.infile) == []):
  print "Error! Could not open infile: ", args.infile
  exit(-1)

sta_cnfg = config.getint('gauge configuration numbers', 'First configuration')
end_cnfg = config.getint('gauge configuration numbers', 'Last configuration')
del_cnfg = config.getint('gauge configuration numbers', \
                                                       'Configuration stepping')
missing_configs = config.get('gauge configuration numbers', \
                                                       'Missing configurations')
# turns missing configs into list of integers
missing_configs = [int(m) for m in missing_configs.split(',')]

if verbose:
  print sta_cnfg
  print end_cnfg
  print del_cnfg
  print missing_configs

T = config.getint('ensemble parameters', 'T')

if verbose:
  print T

# gamma structure wich shall be averaged. Last entry of gamma must contain all
# names in LaTeX compatible notation for plot labels
gamma_dictionary = {
'gamma_i' :   [1, 2, 3, \
              ['\gamma_1', '\gamma_2', '\gamma_3', '\gamma_i']],
'gamma_0i' :  [10, 11, 12, \
              ['\gamma_0\gamma_1', '\gamma_0\gamma_2', '\gamma_0\gamma_3', \
               '\gamma_0\gamma_i']],
'gamma_50i' : [13, 14, 15, \
              ['\gamma_5\gamma_0\gamma_1', '\gamma_5\gamma_0\gamma_2', \
               '\gamma_5\gamma_0\gamma_3', '\gamma_5\gamma_0\gamma_i']],
'gamma_5' :   [5, ['\gamma_5']]
}

p_max = config.getint('gevp parameters', 'p_max')
p_cm = config.getint('gevp parameters', 'p_cm')
p_cm_max = np.asarray([4,5,6,7,4], dtype=int)[p_cm]
gamma_input = config.get('gevp parameters', 'Dirac structure')
# translates list of names for gamma structures to indices used in contraction 
# code
gamma_input = gamma_input.replace(" ", "").split(',')
gammas = [gamma_dictionary[g] for g in gamma_input]

if verbose:
  print p_max
  print p_cm
  print p_cm_max
  print gamma_input

diagrams = config.get('contraction details', 'diagram')
diagrams = diagrams.replace(" ", "").split(',')
directories = config.get('contraction details', 'directory')
directories = directories.replace(" ", "").split(',')

if verbose:
  print diagrams
  print directories 

################################################################################
# Main #########################################################################

def main():


#  for p_cm in p:
  for diagram, directory in zip(diagrams, directories):
    read.ensembles(sta_cnfg, end_cnfg, del_cnfg, diagram, p_cm, p_cm_max, \
                   p_max, gammas, T, directory, missing_configs, verbose)
#    for diagram in diagrams_for_wick:
#      wick.rho_2pt(p_cm, diagram, verbose)

#    for diagram, directory in zip(diagrams, directories):
#      subduce.ensembles()

if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    pass


