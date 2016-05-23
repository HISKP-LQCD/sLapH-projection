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

gevp = ['g0gi', (1,1), (2,2)]
################################################################################
# Functions ####################################################################

################################################################################
# data gevp_row x gevp_col x T x nb_boot
def symmetrize(data, sinh):
  T = data.shape[-2]
  print 'T = ', T
  if sinh.shape != data.shape[:2]:
    print 'in symmetrize: sinh must contain the correct sign for every ' \
          'correlator'
    exit(0)
  if data.ndim != 4:
    print 'in symmetrize: data must have 4 dimensions'
    exit(0)
  sym = np.zeros((data.shape[:2]) + (T/2, data.shape[-1]))
  # loop over gevp matrix elements
  for mat_el in np.ndindex(data.shape[:2]):
    for t in range(1,T/2):
      sym[mat_el+(t,)] = data[mat_el+(t,)] + sinh[mat_el] * data[mat_el+((T-t),)]
    sym[mat_el+(0,)] = data[mat_el+(0,)]

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

################################################################################
# reading data #################################################################

################################################################################

# 2pt subduced + averaged correlators
filename = '../2pt/bootdata/p%1i/C20_p%1i_subduced_avg_rows.npy' % (p, p)
data_2pt = np.load(filename)

filename = '../2pt/bootdata/p%1i/C20_p%1i_subduced_avg_rows_qn.npy' % (p, p)
qn_2pt = np.load(filename)
if ( (qn_2pt.shape[0] != data_2pt.shape[0]) ):
  print 'Bootstrapped operators do not aggree with expected operators'
  exit(0)
print qn_2pt.shape

# 3pt subduced + averaged correlators
filename = '../3pt/bootdata/p%1i/C3+_p%1i_subduced_avg_rows.npy' % (p, p)
data_3pt = np.load(filename)

filename = '../3pt/bootdata/p%1i/C3+_p%1i_subduced_avg_rows_qn.npy' % (p, p)
qn_3pt = np.load(filename)
if ( (qn_3pt.shape[0] != data_3pt.shape[0]) ):
  print 'Bootstrapped operators do not aggree with expected operators'
  exit(0)
print qn_3pt.shape

# 4pt subduced + averaged correlators
filename = '../4pt/bootdata/p%1i/C4_p%1i_subduced_avg_rows.npy' % (p, p)
data_4pt = np.load(filename)

filename = '../4pt/bootdata/p%1i/C4_p%1i_subduced_avg_rows_qn.npy' % (p, p)
qn_4pt = np.load(filename)
if ( (qn_4pt.shape[0] != data_4pt.shape[0]) ):
  print 'Bootstrapped operators do not aggree with expected operators'
  exit(0)
print qn_4pt.shape
print ' '

################################################################################
#GEVP ##########################################################################

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
sinh = np.asarray([[+1, -1, -1, +1, +1], \
                   [-1, +1, +1, -1, -1], \
                   [-1, +1, +1, -1, -1], \
                   [+1, -1, -1, +1, +1], \
                   [+1, -1, -1, +1, +1]])

sym = []

mask = np.empty(qn.shape[:-1], dtype=np.int)
# TODO: automize slicing by giving list of desired quantum numbers
#for i in enumerate(qn):
for i in np.ndindex(mask.shape):
  mask[i] = False if np.sum(list(qn[i]).count(g) for g in gevp) == 2 else True
  # only supported in numpy 1.9.1 or higher
  #  values, counts = np.unique(qn, return_counts=True)

sinh_for_gevp = np.asarray([sinh[i] for i in np.ndindex(mask.shape[1:]) \
                         if not mask[(0,) + i]]).reshape((len(gevp), len(gevp)))
data_for_gevp = np.asarray([data[i] for i in np.ndindex(mask.shape) \
                         if not mask[i]]).reshape( \
                         (data.shape[0], len(gevp), len(gevp)) + data.shape[3:])
qn_for_gevp = np.asarray([qn[i] for i in np.ndindex(mask.shape) \
                         if not mask[i]]).reshape( \
                           (qn.shape[0], len(gevp), len(gevp)) + qn.shape[3:])

print sinh_for_gevp.shape
print data_for_gevp.shape
print qn_for_gevp.shape

for irrep in data_for_gevp:

  sym.append(symmetrize(irrep, sinh_for_gevp))

sym = np.asarray(sym)
print sym.shape

# interface to the code. For gevp desirable to have matrix as last dimension
sym = np.rollaxis(sym, 0, sym.ndim)
sym = sym.T
sym = np.swapaxes(sym, -2, -1)
print sym.shape

# TODO: maybe there is a more elegant solution featuring np.unique (without the
# sorting)
qn_for_gevp = np.swapaxes(qn_for_gevp.diagonal(axis1=-3, axis2=-2), -2, -1)
qn_for_gevp = np.delete(qn_for_gevp, [0,2], -1)
print qn_for_gevp.shape

################################################################################
# Write data to disc ###########################################################

################################################################################
utils.ensure_dir('./bootdata')
utils.ensure_dir('./bootdata/p%1i' % p)
## write all (anti-)symmetrized gevp eigenvalues
path = './bootdata/p%1i/Crho_p%1i_sym' % (p, p)
np.save(path, sym)
path = './bootdata/p%1i/Crho_p%1i_sym_qn' % (p, p)
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




