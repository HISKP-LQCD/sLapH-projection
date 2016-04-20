#!/usr/bin/python

import numpy as np
import itertools as it
import cmath

import operator
import collections

import clebsch_gordan_2pt as cg_2pt
import clebsch_gordan_4pt as cg_4pt
import utils as utils

# loop over irrep
# loop over row
# loop over qn
# if qn belongs to row: 
# add correlator to subduced

p = 0
p_max = 4

diagram = 'C3+'

# Operators entering the Gevp. Last entry must contain the name in LaTeX 
# compatible notation for plot labels
gamma_i =   [1, 2, 3, '\gamma_i']
gamma_0i =  [10, 11, 12, '\gamma_0\gamma_i']
gamma_50i = [15, 14, 13, '\gamma_5\gamma_0\gamma_i']

gammas = [gamma_i, gamma_0i, gamma_50i]
gamma_for_filenames = ['gi', 'g0gi', 'g5g0gi']

################################################################################
# read original data and call bootrap procedure

path = './readdata/p%1i/%s_p%1i.npy' % (p, diagram, p)
data = np.load(path)
path = './readdata/p%1i/%s_p%1i_quantum_numbers.npy' % (p, diagram, p)
qn_data = np.load(path)
if ( (qn_data.shape[0] != data.shape[0])):
  print '\tBootstrapped operators do not aggree with expected operators'
  exit(0)

################################################################################
# From analytic calculation: Irreps contributing to momenta 0 <= p <= 4

def scalar_mul(x, y):
  return sum(it.imap(operator.mul, x, y))

def abs2(x):
  if isinstance(x, collections.Iterator):
    x = list(x)
  return scalar_mul(x, x)


def unique_everseen(iterable, key=None):
  "List unique elements, preserving order. Remember all elements ever seen."
  # unique_everseen('AAAABBBCCDAABBB') --> A B C D
  # unique_everseen('ABBCcAD', str.lower) --> A B C D
  seen = set()
  seen_add = seen.add
  if key is None:
    for element in it.ifilterfalse(seen.__contains__, iterable):
      seen_add(element)
      yield element
  else:
    for element in iterable:
      k = key(element)
      if k not in seen:
        seen_add(k)
        yield element
  
lookup_p3 = it.ifilter(lambda x: abs2(x) <= p_max, \
                       it.product(range(-p_max, p_max+1), repeat=3))

lookup_p = list(it.ifilterfalse(lambda x: x[0] < x[1], \
    unique_everseen(it.starmap(lambda x, y: (abs2(x), abs2(y)), \
    it.ifilter(lambda (x,y): abs2(it.imap(operator.add, x, y)) == p \
               # for zero center of mass momentum omit the case were both 
               # particles are at rest (s-wave)
               and not (p == 0 and tuple(x) == tuple(y) ) \
    ,it.product(lookup_p3, repeat=2)) \
      )) ))
#TODO: for p != 0 there are equivalent combinations with p1 <-> p2
#TODO: critical for moving frames only group by summed 3-momentum at source and sink
#for d in lookup_p:
#  print d

if p in [0]:
  irreps_2pt = [['T1']]
  irreps_4pt = [['T1']]
else:
  irreps_2pt = [[]]
  irreps_4pt = [[]]


# get factors for the desired irreps
for i in irreps_2pt[-1]:
  irreps_2pt.insert(-1,cg_2pt.coefficients(i))
for i in irreps_4pt[-1]:
  irreps_4pt.insert(-1,cg_4pt.coefficients(i))
if len(irreps_4pt) != len(irreps_2pt):
  print 'in subduction.py: irrep for 2pt and 4pt functions contain ' \
        'different number of rows'

correlator = []
qn_subduced = []
for i, (irrep_so, irrep_si) in enumerate(zip(irreps_4pt[:-1], irreps_2pt[:-1])):
  correlator_irrep = []
  qn_irrep = []
  for gamma in gammas:
    correlator_gamma = []
    qn_gamma = []
    for so_mom in lookup_p:
      correlator_mom = []
      qn_mom = []
      for row in range(len(irrep_so)):
    
        correlator_row = np.zeros((0, ) + data[0].shape, dtype=np.double)
        qn_row = []
    
        for so_3mom in irrep_so[row]:
          for si_3mom in irrep_si[row]:
            if not (((np.dot(so_3mom[0], so_3mom[0]), \
                                    np.dot(so_3mom[1], so_3mom[1])) == so_mom) \
                   and (np.dot(si_3mom[0], si_3mom[0]) == p)):
              continue
    
            subduced = np.zeros((1,) + data[0].shape)
    
            for op, qn in enumerate(qn_data):
              if not ((np.array_equal(so_3mom[0], qn[0]) and \
                                                     np.array_equal(so_3mom[1], qn[6])) \
                     and (np.array_equal((-1)*si_3mom[0], qn[3]))):
                continue
              for g_si in range(0,3):
                if not (gamma[g_si] == qn[5]):
                  continue
    
                factor = so_3mom[-1] * np.conj(si_3mom[g_si+1])
                if factor == 0:
                  continue
                subduced[0] = subduced[0] + factor.real * data[op].real + \
                                                         factor.imag * data[op].imag
            if(subduced.any() != 0):
    #          if verbose:
    #            print '\tinto momentum (%i,%i,%i)' % (el[0][0], el[0][1], el[0][2])
    #            print ' '
            # if correlator is 0, still stack subduced
              correlator_row = np.vstack((correlator_row, subduced))
#            else:
#              correlator_row = np.vstack((correlator_row, (-1)*np.zeros_like(subduced) ))
              qn_row.append([ so_3mom[0], so_3mom[1], si_3mom[0], \
                                          so_mom, p, 5, gamma[g_si], irreps_4pt[-1][i] ])
                
        correlator_mom.append(correlator_row)
        qn_mom.append(np.asarray(qn_row))
      correlator_mom = np.asarray(correlator_mom)
      if(np.any(correlator_mom != 0)):
        correlator_gamma.append(correlator_mom)
        qn_gamma.append(np.asarray(qn_mom))
    correlator_irrep.append(correlator_gamma)
    qn_irrep.append(np.asarray(qn_gamma))
  correlator.append(np.asarray(correlator_irrep))
  qn_subduced.append(np.asarray(qn_irrep))

correlator = np.asarray(correlator)
qn_subduced = np.asarray(qn_subduced)

print correlator.shape
print qn_subduced.shape

#for i in range(correlator.shape[0]):
#  for j in range(correlator[i].shape[0]):
#    for k in range(correlator[i,j].shape[0]):
#      for l in range(correlator[i,j,k].shape[0]):
#        print i, j, k, l, correlator[i,j,k,l].shape

utils.ensure_dir('./readdata')
utils.ensure_dir('./readdata/p%1i/' % p)

################################################################################
# write data to disc

# write all subduced correlators
path = './readdata/p%1i/%s_p%1i_single_subduced' % (p, diagram, p)
np.save(path, correlator)
path = './readdata/p%1i/%s_p%1i_single_subduced_quantum_numbers' % (p, diagram, p)
np.save(path, qn_subduced)
 






