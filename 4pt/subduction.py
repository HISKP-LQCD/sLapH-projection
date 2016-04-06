#!/usr/bin/python

import numpy as np
import cmath

import clebsch_gordan as cg
import utils as utils

# loop over irrep
# loop over row
# loop over qn
# if qn belongs to row: 
# add correlator to subduced

p = 0

################################################################################
# read original data and call bootrap procedure

path = './readdata/p%1i/%s_p%1i.npy' % (p, 'C4', p)
data = np.load(path)
path = './readdata/p%1i/%s_p%1i_quantum_numbers.npy' % (p, 'C4', p)
qn_data = np.load(path)
if ( (qn_data.shape[0] != data.shape[0])):
  print '\tBootstrapped operators do not aggree with expected operators'
  exit(0)

################################################################################
# From analytic calculation: Irreps contributing to momenta 0 <= p <= 4

if p in [0]:
  irreps = [['T1']]

  # get factors for the desired irreps
  for i in irreps[-1]:
    irreps.insert(-1,cg.coefficients(i))

correlator = []
quantum_numbers = []
for i, irrep in enumerate(irreps[:-1]):
  correlator_irrep = []
  qn_irrep = []
  for row in irrep:

    correlator_row = np.zeros((0, ) + data[0].shape, dtype=np.double)
    qn_row = []

    for so in row:
      for si in row:

        subduced = np.zeros((1,) + data[0].shape)

        for op, qn in enumerate(qn_data):
          if not ((np.array_equal(so[0], qn[0]) and \
                                                 np.array_equal(so[1], qn[3])) \
                  and (np.array_equal((-1)*si[0], qn[6]) and \
                                            np.array_equal((-1)*si[1], qn[9]))):
            continue

          factor = so[-1] * np.conj(si[-1])
          if factor == 0:
            continue
          subduced[0] = subduced[0] + factor.real * data[op].real + \
                                                     factor.imag * data[op].imag
        if(subduced.any() != 0):
#          if verbose:
#            print '\tinto momentum (%i,%i,%i)' % (el[0][0], el[0][1], el[0][2])
#            print ' '
          correlator_row = np.vstack((correlator_row, subduced))
          qn_row.append([so[0], so[1], si[0], si[1], 5, 5, irreps[-1][i] ])
            
    correlator_irrep.append(correlator_row)
    qn_irrep.append(np.asarray(qn_row))
  correlator.append(np.asarray(correlator_irrep))
  quantum_numbers.append(np.asarray(qn_irrep))

correlator = np.asarray(correlator)
quantum_numbers = np.asarray(quantum_numbers)

print correlator.shape
print quantum_numbers.shape






