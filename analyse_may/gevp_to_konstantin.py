#!/usr/bin/python

import math
import numpy as np
import numpy.ma as ma
import itertools as it

import sys
#sys.stdout = open('./irrep_and_gevp.out', 'w')

import gevp
import utils

################################################################################
# some global variables ########################################################

################################################################################
p = range(5) # momenta to analyse

gevp = [[['gi', (2,2), (1,1)]], \
        [['gi', (4,1), (3,2), (2,1), (1,0)], \
         ['gi', (3,2), (2,1)]], \
        [['gi', (4,2), (3,1), (2,0)], \
         ['gi', (4,2), (2,2), (1,1)],
         ['gi', (3,1), (2,2)]], \
        [['gi', (4,3), (3,0), (2,1)]], \
        [['gi', (4,0)]]]

#        [['gi', (4,2), (3,1), (2,0), (2,2), (1,1)], \
##################################################################################
for p_cm in p:
  print 'building gevp for p_cm = %i\n' % p_cm
  ################################################################################
  # reading data #################################################################
  
  ################################################################################
  
  # 2pt subduced + averaged correlators
  filename = '../2pt/readdata/p%1i/C20_p%1i_subduced_avg_rows.npy' % (p_cm, p_cm)
  data_2pt = np.load(filename)
  
  filename = '../2pt/readdata/p%1i/C20_p%1i_subduced_avg_rows_qn.npy' % (p_cm, p_cm)
  qn_2pt = np.load(filename)
  if ( (qn_2pt.shape[0] != data_2pt.shape[0]) ):
    print 'Operators do not aggree with expected operators'
    exit(0)
  print '\tqn_2pt: ', qn_2pt.shape
  
  # 3pt subduced + averaged correlators
  filename = '../3pt/readdata/p%1i/C3+_p%1i_subduced_avg_rows.npy' % (p_cm, p_cm)
  data_3pt = np.load(filename)
  
  filename = '../3pt/readdata/p%1i/C3+_p%1i_subduced_avg_rows_qn.npy' % (p_cm, p_cm)
  qn_3pt = np.load(filename)
  if ( (qn_3pt.shape[0] != data_3pt.shape[0]) ):
    print 'Operators do not aggree with expected operators'
    exit(0)
  print '\tqn_3pt: ', qn_3pt.shape
  
  # 4pt subduced + averaged correlators
  filename = '../4pt/readdata/p%1i/C4_p%1i_subduced_avg_rows.npy' % (p_cm, p_cm)
  data_4pt = np.load(filename)
  
  filename = '../4pt/readdata/p%1i/C4_p%1i_subduced_avg_rows_qn.npy' % (p_cm, p_cm)
  qn_4pt = np.load(filename)
  if ( (qn_4pt.shape[0] != data_4pt.shape[0]) ):
    print 'Operators do not aggree with expected operators'
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
   
    mask = np.empty(qn.shape[:-1], dtype=np.int)
    for j in np.ndindex(mask.shape):
      values, counts = np.unique(qn[j], return_counts=True)
      dictionary = {v : c for v,c in zip(values,counts)} 
      mask[j] = False if np.sum(dictionary.get(g, 0) for g in gevp[p_cm][i]) == 2 else True
    data_for_gevp = np.asarray([data[j] for j in np.ndindex(mask.shape) \
                             if not mask[j]]).reshape( \
                             (len(gevp[p_cm][i]), len(gevp[p_cm][i])) + data.shape[2:])
    print data_for_gevp.shape

    qn_for_gevp = np.asarray([qn[j] for j in np.ndindex(mask.shape) \
                             if not mask[j]]).reshape( \
                               (len(gevp[p_cm][i]), len(gevp[p_cm][i])) + qn.shape[2:])
    qn_for_gevp = qn_for_gevp.diagonal().T
    print qn_for_gevp.shape

    ################################################################################
    # Write data to disc ###########################################################
    
    ################################################################################
    utils.ensure_dir('./readdata')
    utils.ensure_dir('./readdata/p%1i' % p_cm)
    utils.ensure_dir('./data')
    utils.ensure_dir('./data/p%1i' % p_cm)
    utils.ensure_dir('./data/p%1i/%s' % (p_cm, qn_for_gevp[0,-1]))
    
    # write the orginal data for all gevp matrix elements as npy
    path = './readdata/p%1i/Crho_p%1i_%s' % (p_cm, p_cm, qn_for_gevp[0,-1])
    np.save(path, data_for_gevp)
    path = './readdata/p%1i/Crho_p%1i_%s_qn' % (p_cm, p_cm, qn_for_gevp[0,-1])
    np.save(path, qn_for_gevp)
    
    ################################################################################
    # write the orginal data for all gevp matrix elements individually as ascii
    for gevp_row, row in enumerate(data_for_gevp):
      for gevp_col, col in enumerate(row):
        path = './data/p%1i/%s/Rho_Gevp_p%1d_%s.%d.%d.dat' % \
                (p_cm, qn_for_gevp[0,-1], p_cm, \
                          qn_for_gevp[0,-1], gevp_row, gevp_col)
        print 'writing to', path
        T = col.shape[0]
    
        # insert zeros as symmetrization partner for timeslices 0 and T/2
        X = np.copy(col)
        X = np.insert(X, T, 0, axis=0)
        X = np.insert(X, T/2+1, 0, axis=0)
        X = np.asarray((X[:(T/2+1),...], X[:(T/2):-1,...])).T
        
        # flatten indices for nb_cnfg and T and insert dummy indices for gamma 
        # and smear
        Y = np.zeros((X.shape[0]*1*4*(T/2+1), 6))
        for c, cnfg in enumerate(range(714,2750,2)):
          if cnfg in [1282]:
            continue
          for gamma in range(1):
            for s, smear in enumerate(range(1,8,2)):
              for t in range(T/2+1):
                Y[t+s*(T/2+1)+c*4*(T/2+1)] = np.asarray([gamma, smear, t, \
                                                        X[c,t,0], X[c,t,1], cnfg])
        print '\t', Y.shape
    
        np.savetxt(path, Y, fmt=['%d','%d','%d','%f','%f','%d'], delimiter=' ', \
                             newline='\n')
    
    # TODO:
    # write the orginal data for all gevp matrix elements individually as binary
    #      path = './binarydata/p%1i/C20_p%1i_E2_%s_%s.dat' % (p, p, \
    #                        gamma_for_filenames[qn_subduced[i,j,0,-3]], \
    #                        gamma_for_filenames[qn_subduced[i,j,0,-2]])
    #      f = open(path, 'wb')
    #      E2 = np.mean(np.vstack((correlator[-1,j], correlator[-2,j])), axis=0)
    #      (E2.swapaxes(-1, -2)).tofile(f)



