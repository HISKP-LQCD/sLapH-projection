#!/usr/bin/python

import math
import numpy as np

import sys
#sys.stdout = open('./irrep_and_gevp.out', 'w')

import gevp
import utils

################################################################################
# some global variables ########################################################

################################################################################
p = 2 # momenta to analyse

################################################################################
# Functions ####################################################################

################################################################################
def symmetrize(data, sinh):
  T = data.shape[1]
  if len(sinh) != data.shape[0]:
    print 'in symmetrize: sinh must contain the correct sign for every ' \
          'correlator'
    exit(0)
  if data.ndim != 3:
    print 'in symmetrize: data must have 3 dimensions'
    exit(0)
  sym = np.zeros((data.shape[0], T/2, data.shape[2]))
  # loop over gevp matrix elements
  for mat_el in range(sym.shape[0]):
    for t in range(1,T/2):
      sym[mat_el,t] = data[mat_el,t] + sinh[mat_el] * data[mat_el,T-t]
    sym[mat_el,0] = data[mat_el, 0]

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
# subduced + averaged correlators
filename = './bootdata/p%1i/C20_p%1i_avg_subduced.npy' % (p, p)
data = np.load(filename)

filename = './bootdata/p%1i/C20_p%1i_avg_subduced_quantum_numbers.npy' % (p, p)
qn_sub_avg = np.load(filename)
if ( (qn_sub_avg.shape[0] != data.shape[0]) ):
  print 'Bootstrapped operators do not aggree with expected operators'
  exit(0)

################################################################################
#GEVP ##########################################################################

#filename = 'bootdata/rho_corr_TP%d_00.npy' % p
#rho_00 = np.load(filename)
#filename = 'bootdata/rho_corr_TP%d_01.npy' % p
#rho_01 = np.load(filename)
#filename = 'bootdata/rho_corr_TP%d_11.npy' % p
#rho_11 = np.load(filename)

# if g0gi is daggered, the order switches. Switching back generates and extra
# minus sign
sinh = [+1, -1, -1, -1, +1, +1, -1, +1, +1]
negative = [1,3]
pc = []
for i in range(data.shape[0]):
#for i in range(0,1):
  for j in range(data[i].shape[0]):
    if j in negative:
      data[i,j] = (2) * data[i,j]
    else:
      data[i,j] = -2 * data[i,j]
 
  sym = symmetrize(data[i], sinh)

  for t0 in range(1,2):
    pc.append(gevp.calculate_gevp(sym, t0))

pc = np.asarray(pc)
print pc.shape
print pc

################################################################################
# Write data to disc ###########################################################

################################################################################
utils.ensure_dir('./bootdata')
utils.ensure_dir('./bootdata/p%1i' % p)

################################################################################
# write all (anti-)symmetrized gevp eigenvalues
path = './bootdata/p%1i/C20_p%1i_sym_gevp' % (p, p)
np.save(path, pc)
path = './bootdata/p%1i/C20_p%1i_sym_gevp_quantum_numbers' % (p, p)
# TODO: save t0 in qn
np.save(path, qn_sub_avg[:,0,-1])




