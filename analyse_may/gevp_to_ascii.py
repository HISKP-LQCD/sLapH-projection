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
p = 0 # momenta to analyse

gevp = ['gi', 'g0gi', 'g5g0gi', (1,1), (2,2)] # desired gevp operators

################################################################################
# Functions ####################################################################

################################################################################
# reading data #################################################################

################################################################################

# 2pt subduced + averaged correlators
filename = '../2pt/readdata/p%1i/C20_p%1i_subduced_avg_rows.npy' % (p, p)
data_2pt = np.load(filename)

filename = '../2pt/readdata/p%1i/C20_p%1i_subduced_avg_rows_qn.npy' % (p, p)
qn_2pt = np.load(filename)
if ( (qn_2pt.shape[0] != data_2pt.shape[0]) ):
  print 'Read operators do not aggree with expected operators'
  exit(0)
print qn_2pt.shape

# 3pt subduced + averaged correlators
filename = '../3pt/readdata/p%1i/C3+_p%1i_subduced_avg_rows.npy' % (p, p)
data_3pt = np.load(filename)

filename = '../3pt/readdata/p%1i/C3+_p%1i_subduced_avg_rows_qn.npy' % (p, p)
qn_3pt = np.load(filename)
if ( (qn_3pt.shape[0] != data_3pt.shape[0]) ):
  print 'Read operators do not aggree with expected operators'
  exit(0)
print qn_3pt.shape

# 4pt subduced + averaged correlators
filename = '../4pt/readdata/p%1i/C4_p%1i_subduced_avg_rows.npy' % (p, p)
data_4pt = np.load(filename)

filename = '../4pt/readdata/p%1i/C4_p%1i_subduced_avg_rows_qn.npy' % (p, p)
qn_4pt = np.load(filename)
if ( (qn_4pt.shape[0] != data_4pt.shape[0]) ):
  print 'Read operators do not aggree with expected operators'
  exit(0)
print qn_4pt.shape
print ' '

################################################################################
# combining the gevp matrix ####################################################

data = np.concatenate( \
        (np.concatenate((data_2pt, np.swapaxes(data_3pt, 1, 2)), axis=2), \
         np.concatenate((data_3pt,             data_4pt),        axis=2)), \
                                                                         axis=1)
qn = np.concatenate( \
      (np.concatenate((qn_2pt, np.swapaxes(qn_3pt, 1, 2)), axis=2), \
       np.concatenate((qn_3pt,             qn_4pt),        axis=2)), axis=1)

print data.shape
print qn.shape

#                    g_i g_0i g_50i [1,1] [2,2]

mask = np.empty(qn.shape[:-1], dtype=np.int)
# TODO: automize slicing by giving list of desired quantum numbers
#for i in enumerate(qn):
for i in np.ndindex(mask.shape):
  mask[i] = False if np.sum(list(qn[i]).count(g) for g in gevp) == 2 else True
  # only supported in numpy 1.9.1 or higher:
  #  values, counts = np.unique(qn, return_counts=True)

data_for_gevp = np.asarray([data[i] for i in np.ndindex(mask.shape) \
                         if not mask[i]]).reshape( \
                         (data.shape[0], len(gevp), len(gevp)) + data.shape[3:])
qn_for_gevp = np.asarray([qn[i] for i in np.ndindex(mask.shape) \
                         if not mask[i]]).reshape( \
                           (qn.shape[0], len(gevp), len(gevp)) + qn.shape[3:])

print data_for_gevp.shape
print qn_for_gevp.shape

################################################################################
# Write data to disc ###########################################################

################################################################################
utils.ensure_dir('./readdata')
utils.ensure_dir('./readdata/p%1i' % p)
utils.ensure_dir('./data')
utils.ensure_dir('./data/p%1i' % p)
for irrep in np.unique(qn_for_gevp[...,-1]):
  utils.ensure_dir('./data/p%1i/%s' % (p, irrep))

# write the orginal data for all gevp matrix elements as npy
path = './readdata/p%1i/Crho_p%1i' % (p, p)
np.save(path, data_for_gevp)
path = './readdata/p%1i/Crho_p%1i_qn' % (p, p)
np.save(path, qn_for_gevp)

################################################################################
# write the orginal data for all gevp matrix elements individually as ascii
for i, irrep in enumerate(data_for_gevp):
  for gevp_row, row in enumerate(irrep):
    for gevp_col, col in enumerate(row):
      path = './data/p%1i/%s/Rho_Gevp_p%1d_%s.%d.%d.dat' % \
              (p, qn_for_gevp[i,gevp_row,gevp_col,-1], p, \
                        qn_for_gevp[i,gevp_row,gevp_col,-1], gevp_row, gevp_col)
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
      for c, cnfg in enumerate(range(714,1215,2)):
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



