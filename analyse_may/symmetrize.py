#!/usr/bin/python

import math
import numpy as np
import numpy.ma as ma
import itertools as it

import sys
#sys.stdout = open('./irrep_and_gevp.out', 'w')

#import gevp
import utils

################################################################################
# some global variables ########################################################

################################################################################
p = range(5) # momenta to analyse

gevp = [[['gi', (1,1), (2,2)]], \
        [['gi', (4,1), (3,2), (2,1), (1,0)], \
         ['gi', (3,2), (2,1)]], \
        [['gi', (4,2), (3,1), (2,0), (2,2), (1,1)], \
         ['gi', (4,2), (2,2), (1,1)],
         ['gi', (3,1), (2,2)]], \
        [['gi', (4,3), (3,0), (2,1)]], \
        [['gi', (4,0)]]]

################################################################################
# Functions ####################################################################

################################################################################
# data gevp_row x gevp_col x T x nb_boot
#def symmetrize(data, sinh):
def symmetrize(data):
  T = data.shape[-2]
  print 'T = ', T
#  if sinh.shape != data.shape[:2]:
#    print 'in symmetrize: sinh must contain the correct sign for every ' \
#          'correlator'
#    exit(0)
  if data.ndim != 4:
    print 'in symmetrize: data must have 4 dimensions'
    exit(0)
  sym = np.zeros((data.shape[:2]) + (T/2+1, data.shape[-1]))
  # loop over gevp matrix elements
  for mat_el in np.ndindex(data.shape[:2]):
    for t in range(1,T/2):
#      sym[mat_el+(t,)] = data[mat_el+(t,)] + sinh[mat_el] * data[mat_el+((T-t),)]
      sym[mat_el+(t,)] = data[mat_el+(t,)] + data[mat_el+((T-t),)]
    sym[mat_el+(0,)] = data[mat_el+(0,)]
    sym[mat_el+(T/2,)] = data[mat_el+(T/2,)]

  return sym

################################################################################
# computes the mean and the error, and writes both out
def mean_error_print(boot, write = 0):
  mean = np.mean(boot, axis=-1)
  err  = np.std(boot, axis=-1)
  if write:
    for t, m, e in zip(range(0, len(mean)), mean, err):
      print t, m, e
  return mean, err

for p_cm in p:
  print 'building gevp for p_cm = %i\n' % p_cm
  ################################################################################
  # reading data #################################################################
  
  ################################################################################
  
  # 2pt subduced + averaged correlators
  filename = '../2pt/bootdata/p%1i/C20_p%1i_subduced_avg_rows.npy' % (p_cm, p_cm)
  data_2pt = np.load(filename)
  
  filename = '../2pt/bootdata/p%1i/C20_p%1i_subduced_avg_rows_qn.npy' % (p_cm, p_cm)
  qn_2pt = np.load(filename)
  if ( (qn_2pt.shape[0] != data_2pt.shape[0]) ):
    print 'Bootstrapped operators do not aggree with expected operators'
    exit(0)
  print '\tqn_2pt: ', qn_2pt.shape
  
  # 3pt subduced + averaged correlators
  filename = '../3pt/bootdata/p%1i/C3+_p%1i_subduced_avg_rows.npy' % (p_cm, p_cm)
  data_3pt = np.load(filename)
  
  filename = '../3pt/bootdata/p%1i/C3+_p%1i_subduced_avg_rows_qn.npy' % (p_cm, p_cm)
  qn_3pt = np.load(filename)
  if ( (qn_3pt.shape[0] != data_3pt.shape[0]) ):
    print 'Bootstrapped operators do not aggree with expected operators'
    exit(0)
  print '\tqn_3pt: ', qn_3pt.shape
  
  # 4pt subduced + averaged correlators
  filename = '../4pt/bootdata/p%1i/C4_p%1i_subduced_avg_rows.npy' % (p_cm, p_cm)
  data_4pt = np.load(filename)
  
  filename = '../4pt/bootdata/p%1i/C4_p%1i_subduced_avg_rows_qn.npy' % (p_cm, p_cm)
  qn_4pt = np.load(filename)
  if ( (qn_4pt.shape[0] != data_4pt.shape[0]) ):
    print 'Bootstrapped operators do not aggree with expected operators'
    exit(0)
  print '\tqn_4pt: ', qn_4pt.shape
  print ' '
  
  ################################################################################
  #GEVP ##########################################################################
  
  for i in range(len(gevp[p_cm])):

    data = np.concatenate( \
            (np.concatenate((data_2pt[i], np.swapaxes(data_3pt[i], 0, 1)), axis=1), \
             np.concatenate((data_3pt[i],             data_4pt[i]),        axis=1)), \
                                                                             axis=0)
    qn = np.concatenate( \
          (np.concatenate((qn_2pt[i], np.swapaxes(qn_3pt[i], 0, 1)), axis=1), \
           np.concatenate((qn_3pt[i],             qn_4pt[i]),        axis=1)), axis=0)
    
    print data.shape
    print qn.shape
    
    #                    g_i g_0i g_50i [1,1] [2,2]
    sinh = np.asarray([[+1, -1, -1, +1, +1], \
                       [-1, +1, +1, -1, -1], \
                       [-1, +1, +1, -1, -1], \
                       [+1, -1, -1, +1, +1], \
                       [+1, -1, -1, +1, +1]])
    
    sym = []

    print qn
    print ' '
    print gevp[p_cm][i]
    
    mask = np.empty(qn.shape[:-1], dtype=np.int)
    # TODO: automize slicing by giving list of desired quantum numbers
    #for i in enumerate(qn):
    for j in np.ndindex(mask.shape):
      values, counts = np.unique(qn[j], return_counts=True)
      dictionary = {v : c for v,c in zip(values,counts)} 
      mask[j] = False if np.sum(dictionary.get(g, 0) for g in gevp[p_cm][i]) == 2 else True
#    sinh_for_gevp = np.asarray([sinh[i] for i in np.ndindex(mask.shape[1:]) \
#                             if not mask[(0,) + i]]).reshape((len(gevp), len(gevp)))
    data_for_gevp = np.asarray([data[j] for j in np.ndindex(mask.shape) \
                             if not mask[j]]).reshape( \
                             (len(gevp[p_cm][i]), len(gevp[p_cm][i])) + data.shape[2:])
#                             (data.shape[0], len(gevp), len(gevp)) + data.shape[3:])
    qn_for_gevp = np.asarray([qn[j] for j in np.ndindex(mask.shape) \
                             if not mask[j]]).reshape( \
                               (len(gevp[p_cm][i]), len(gevp[p_cm][i])) + qn.shape[2:])
#                               (qn.shape[0], len(gevp), len(gevp)) + qn.shape[3:])
    
#    print sinh_for_gevp.shape
    print data_for_gevp.shape
    print qn_for_gevp.shape
    
    #  sym.append(symmetrize(irrep, sinh_for_gevp))
    sym.append(symmetrize(data_for_gevp))
    
    sym = np.asarray(sym)
    print sym.shape
    
    # interface to the code. For gevp desirable to have matrix as last dimension
    sym = np.rollaxis(sym, 0, sym.ndim)
    sym = sym.T
    sym = np.swapaxes(sym, -2, -1)
    print 'sym', sym.shape
    
    # TODO: maybe there is a more elegant solution featuring np.unique (without the
    # sorting)
    qn_for_gevp = qn_for_gevp.diagonal().T
    print 'qn', qn_for_gevp.shape
    
    ################################################################################
    # Write data to disc ###########################################################
    
    ################################################################################
    utils.ensure_dir('./bootdata')
    utils.ensure_dir('./bootdata/p%1i' % p_cm)
    ## write all (anti-)symmetrized gevp eigenvalues
    path = './bootdata/p%1i/Crho_p%1i_%s_sym' % (p_cm, p_cm, qn_for_gevp[0,-1])
    np.save(path, sym)
    path = './bootdata/p%1i/Crho_p%1i_%s_sym_qn' % (p_cm, p_cm, qn_for_gevp[0,-1])
    np.save(path, qn_for_gevp)
    ## TODO: save t0 in qn
  
  #################################################################################
  ## write all (anti-)symmetrized gevp eigenvalues
  #path = './bootdata/p%1i/p%1i_sym_gevp' % (p, p)
  #np.save(path, pc)
  #path = './bootdata/p%1i/p%1i_sym_gevp_quantum_numbers' % (p, p)
  ## TODO: save t0 in qn
  ##np.save(path, np.concatenate((qn_2pt[:,0,0,-1], qn_4pt[:,1,1,-1]), axis=1).reshape(1,2))
  ##qn = np.concatenate((qn_2pt[0,...,-1].diagonal , qn_4pt[0,...,-1].diagonal), axis=1).reshape(1,5)
  #qn = np.concatenate((qn_2pt[0,...,-1].diagonal(), qn_4pt[0,...,-1].diagonal())).reshape(1,5)
  ##print qn_2pt[0,...,-1].diagonal().shape
  ##print qn_4pt[0,...,-1].diagonal().shape
  #print qn.shape
  #np.save(path, qn)




