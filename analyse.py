#!/hiskp2/werner/libraries/Python-2.7.12/python
import argparse
import ConfigParser

import numpy as np

import raw_data
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
  if(missing_configs == ''):
    missing_configs = []
  else:
    missing_configs = [int(m) for m in missing_configs.split(',')]
  
  if verbose:
    print '#################################################################'\
                                                               '###############'
    print 'Reading infile'

    print sta_cnfg
    print end_cnfg
    print del_cnfg
    print missing_configs
  
  ensemble = config.get('ensemble parameters', 'Ensemble Name')
  T = config.getint('ensemble parameters', 'T')
  
  if verbose:
    print ensemble
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
  
  diagrams = config.get('contraction details', 'Diagram')
  diagrams = diagrams.replace(" ", "").split(',')
  directories = config.get('contraction details', 'Input Path')
  directories = directories.replace(" ", "").replace("\n", "")
  directories = directories.split(',')
  # use the same directory for all diagrams if only one is given
  if(len(directories) == 1):
    directories = directories*len(diagrams)
  
  if verbose:
    print diagrams
    print directories 

  outpath = config.get('other parameters', 'Output Path')

  if verbose:
    print outpath

  ############################################################################## 
  # Main
  for p_cm in p:
    if verbose:
      print '################################################################'\
                                                              '################'
      print 'p_cm = ', p_cm

    ############################################################################ 
    # read diagrams
    data = {}
    lookup_qn = {}
    for diagram, directory in zip(diagrams, directories):

      if verbose:
        print '\treading data for %s' % (diagram)
      lookup_cnfg = raw_data.set_lookup_cnfg(sta_cnfg, end_cnfg, del_cnfg, \
                                                       missing_configs, verbose)
      # set up lookup table for quantum numbers
      if p_cm == 0:
        p_max = 2
      # for moving frames, sum of individual component's absolute value (i.e.
      # total kindetic energy) might be larger than center of mass absolute 
      # value. Modify the cutoff accordingly.
#      p_cm_max = np.asarray([4,5,6,7,4], dtype=int)[p_cm]
      # TODO: that needs to be refactored when going to a larger operator basis
      lookup_qn[diagram] = raw_data.set_lookup_qn(diagram, p_cm, p_max, 
                                                                gammas, verbose)
  
      data[diagram] = raw_data.read(lookup_cnfg, lookup_qn, diagram, T, 
                                                             directory, verbose)
      # write data
      path = '%s/%s/0_raw-data/' % (outpath, ensemble)
      filename = '%s_p%1i.h5' % (diagram, p_cm)
      utils.write_hdf5_correlators(path, filename, data, 'data', verbose)
      utils.write_hdf5_correlators(path, filename, lookup_qn, 'qn', verbose)
 
    ############################################################################
    # Subduction

    lookup_irreps = subduction.set_lookup_irreps(p_cm)

    subduced_data = {}
    lookup_qn_irrep = {}
    for diagram in diagrams:

      print 'subducing data for %s' % diagram
      for irrep in lookup_irreps:

        print '\tsubducing into %s' % irrep

        lookup_qn_irrep[(diagram, irrep)] = subduction.set_lookup_qn_irrep( \
                                              lookup_qn[diagram], diagram, \
                                                   gammas, p_cm, irrep, verbose)
        subduced_data[(diagram, irrep)] = subduction.ensembles(data[diagram], \
                                                       lookup_qn_irrep[diagram])
        # write data to disc
        path = '%s/%s/1_subduced-data/' % (outpath, ensemble)
        filename = '/%s_p%1i_%s.h5' % (diagram, p_cm, irrep)
        utils.write_hdf5_correlators(path, filename, subduced_data, 'data', \
                                                                        verbose)

#        ########################################################################
#        # sum over gamma structures. 
#        # Only real part is physically relevant at that point
#        subduced = subduced.apply(np.real).sum(level=[0,1,2,3,5])
#        # sum over equivalent momenta
#        subduced = subduced.sum(level=[0,1,2])
#        # average over rows
#        subduced = subduced.mean(level=[0,1])
#
#        ##############################################################################
#        # write data to disc
#        path = './readdata/%s_p%1i_%s_avg.h5' % (diagram, p_cm, irrep)
#        utils.ensure_dir('./readdata')
#        utils.write_hdf5_correlators(path, subduced, None)

    ############################################################################ 
    # wick contraction
    # TODO: make wick contractions return data and quantum numbers as well
    # TODO: factor out calculation of correlators and put into correlators loop

    wick.set_lookup_correlators()
    for irrep in lookup_irreps:

       correlators = wick.rho(p_cm, diagrams, verbose)

#    ############################################################################ 
#    # Gevp Construction
#    for irrep in lookup_irreps:
#      gevp = setup_gevp.build_gevp(p_cm, irrep, verbose)

if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    pass


