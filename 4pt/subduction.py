#!/usr/bin/python

import numpy as np
import itertools as it
import cmath

import operator
import collections

import clebsch_gordan as cg
import utils as utils

# loop over irrep
# loop over row
# loop over qn
# if qn belongs to row: 
# add correlator to subduced

p = 0
p_max = 4

diagram = 'C4+B'

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
  irreps = [['T1']]
else:
  irreps = [[]]

# get factors for the desired irreps
for i in irreps[-1]:
  irreps.insert(-1,cg.coefficients(i))

correlator = []
qn_subduced = []
for i, irrep in enumerate(irreps[:-1]):
  correlator_irrep = []
  qn_irrep = []
  for so_mom in lookup_p:
    for si_mom in lookup_p:
      correlator_mom = []
      qn_mom = []
      for row in irrep:
    
        correlator_row = np.zeros((0, ) + data[0].shape, dtype=np.double)
        qn_row = []
    
        for so_3mom in row:
          for si_3mom in row:
            if not (((np.dot(so_3mom[0], so_3mom[0]), np.dot(so_3mom[1], so_3mom[1])) == so_mom) \
                   and ((np.dot(si_3mom[0], si_3mom[0]), np.dot(si_3mom[1], si_3mom[1])) == si_mom)):
              continue
  
            subduced = np.zeros((1,) + data[0].shape)
    
            for op, qn in enumerate(qn_data):
              if not ((np.array_equal(so_3mom[0], qn[0]) and \
                                                     np.array_equal(so_3mom[1], qn[3])) \
                      and (np.array_equal((-1)*si_3mom[0], qn[6]) and \
                                                np.array_equal((-1)*si_3mom[1], qn[9]))):
                continue
   
              factor = so_3mom[-1] * np.conj(si_3mom[-1])
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
              qn_row.append([ so_3mom[0], so_3mom[1], si_3mom[0], si_3mom[1], so_mom, si_mom, 5, 5, irreps[-1][i] ])
                
        correlator_mom.append(correlator_row)
        qn_mom.append(np.asarray(qn_row))
      correlator_mom = np.asarray(correlator_mom)
      if(np.any(correlator_mom != 0)):
        correlator_irrep.append(correlator_mom)
        qn_irrep.append(np.asarray(qn_mom))
  correlator.append(np.asarray(correlator_irrep))
  qn_subduced.append(np.asarray(qn_irrep))

correlator = np.asarray(correlator)
qn_subduced = np.asarray(qn_subduced)

print correlator.shape
print qn_subduced.shape

#for i in correlator:
#  for j in i:
#    for k in j:
#      print k.shape

utils.ensure_dir('./readdata')
utils.ensure_dir('./readdata/p%1i/' % p)

################################################################################
# write data to disc

# write all subduced correlators
path = './readdata/p%1i/%s_p%1i_single_subduced' % (p, diagram, p)
np.save(path, correlator)
path = './readdata/p%1i/%s_p%1i_single_subduced_quantum_numbers' % (p, diagram, p)
np.save(path, qn_subduced)
 






