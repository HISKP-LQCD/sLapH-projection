#!/hiskp2/werner/libraries/Python-2.7.12/python
import argparse
import ConfigParser

import numpy as np

import read
import utils
import wick
import subduction
import setup_gevp

################################################################################
# Main #########################################################################

def main():

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
    print '################################################################################'
    print 'Reading infile'

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
  p = config.get('gevp parameters', 'p_cm')
  p = [int(k) for k in p.split(',')]
  gamma_input = config.get('gevp parameters', 'Dirac structure')
  # translates list of names for gamma structures to indices used in contraction 
  # code
  gamma_input = gamma_input.replace(" ", "").split(',')
  gammas = [gamma_dictionary[g] for g in gamma_input]
  
  if verbose:
    print p_max
    print p
    print gamma_input
  
  diagrams = config.get('contraction details', 'diagram')
  diagrams = diagrams.replace(" ", "").split(',')
  directories = config.get('contraction details', 'directory')
  directories = directories.replace(" ", "").replace("\n", "")
  directories = directories.split(',')
  
  if verbose:
    print diagrams
    print directories 

  ############################################################################## 
  # Main
  for p_cm in p:
    print 'p_cm = ', p_cm

    ############################################################################ 
    # read diagrams
#    if verbose:
#      print '################################################################################'
#      print 'Start reading correlation function diagrams'
#    for diagram, directory in zip(diagrams, directories):
#      lookup_cnfg = read.set_lookup_cnfg(sta_cnfg, end_cnfg, del_cnfg, \
#                                                       missing_configs, verbose)
#      # set up lookup table for quantum numbers
#      if p_cm == 0:
#        p_max = 2
#      # TODO: I don't now what this was good for
##      p_cm_max = np.asarray([4,5,6,7,4], dtype=int)[p_cm]
#      lookup_qn = read.set_lookup_qn(diagram, p_cm, p_max, gammas, verbose)
#  
#      data = read.ensembles(lookup_cnfg, lookup_qn, diagram, T, directory, 
#                                                                        verbose)
#      # write data
#      path = './readdata/%s_p%1i.h5' % (diagram, p_cm)
#      utils.ensure_dir('./readdata')
#      utils.write_hdf5_correlators(path, data, lookup_qn)
#  
    ############################################################################ 
    # wick contraction
    # TODO: make wick contractions return data and quantum numbers as well
    # TODO: factor out calculation of correlators and put into correlators loop
    if verbose:
      print '################################################################################'
      print 'Starting Wick contractions'
    correlators = wick.rho(p_cm, diagrams, verbose)

    ############################################################################ 
    # Subduction
    # TODO: Instead get data, lookup_qn from wick.rho() and reorder loops
    if verbose:
      print '################################################################################'
      print 'Starting Subduction'

    lookup_irreps = subduction.set_lookup_irreps(p_cm)
    for correlator in correlators:
      contracted_data, lookup_qn = utils.read_hdf5_correlators( \
                                     'readdata/%s_p%1i.h5' % (correlator, p_cm))
      contracted_data.columns.name = 'index'
      for irrep in lookup_irreps:
        lookup_qn_irrep = subduction.set_lookup_qn_irrep(lookup_qn, correlator,\
                                                   gammas, p_cm, irrep, verbose)
        subduced_data = subduction.ensembles(contracted_data, lookup_qn_irrep, \
                                        p_cm, correlator, p_max, irrep, verbose)
    ############################################################################ 
    # Gevp Construction
    for irrep in lookup_irreps:
      gevp = setup_gevp.build_gevp(p_cm, irrep, verbose)
      print gevp

if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    pass


