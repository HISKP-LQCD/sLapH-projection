#!/usr/bin/python

import numpy as np
import itertools as it
import cmath

import operator
import collections

import clebsch_gordan as cg
import utils as utils

p = range(5)
p_max = 4

diagram = 'C4'

################################################################################
# From analytic calculation: Irreps contributing to momenta 0 <= p <= 4

def scalar_mul(x, y):
  return sum(it.imap(operator.mul, x, y))

def abs2(x):
  if isinstance(x, collections.Iterator):
    x = list(x)
  return scalar_mul(x, x)

# taken from https://docs.python.org/2/library/itertools.html#module-itertools
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
  

###############################################################################
# From analytic calculation: Irreps contributing to momenta 0 <= p <= 4

for p_cm in p:
  print '\nsubducing p = %i' % p_cm

  ################################################################################
  # read original data 
  
  path = './readdata/p%1i/%s_p%1i.npy' % (p_cm, diagram, p_cm)
  data = np.load(path)
  path = './readdata/p%1i/%s_p%1i_qn.npy' % (p_cm, diagram, p_cm)
  qn_data = np.load(path)
  if ( (qn_data.shape[0] != data.shape[0])):
    print '\tRead operators do not aggree with expected operators'
    exit(0)

  if p_cm in [0]:
    irreps = [['T1']]
  elif p_cm in [1,3,4]:
    irreps = [['A1', 'E2']]
  elif p_cm in [2]:
    irreps = [['A1', 'B1', 'B2']]
  else:
    irreps = [[]]
  # get factors for the desired irreps
  for i in irreps[-1]:
    irreps.insert(-1,cg.coefficients(i))
 
  lookup_p3 = it.ifilter(lambda x: abs2(x) <= p_max, \
                         it.product(range(-p_max, p_max+1), repeat=3))
  lookup_p = list(it.ifilterfalse(lambda x: x[0] < x[1], \
      unique_everseen(it.starmap(lambda x, y: (abs2(x), abs2(y)), \
      it.ifilter(lambda (x,y): abs2(it.imap(operator.add, x, y)) == p_cm \
                 # for zero center of mass momentum omit the case were both 
                 # particles are at rest (s-wave)
                 and not (p_cm == 0 and tuple(x) == tuple(y) ) \
      ,it.product(lookup_p3, repeat=2)) \
        )) ))
 
  correlator = []
  qn_subduced = []
  # loop over irreducible representations subducing into p_cm
  for i, irrep in enumerate(irreps[:-1]):
    correlator_irrep = []
    qn_irrep = []
    # loop over absolute values of momenta at source and sink [k1, k2]
    for gevp_row in lookup_p:
      correlator_gevp_row = []
      qn_gevp_row = []
      for gevp_col in lookup_p:
        correlator_gevp_col = []
        qn_gevp_col = []
        # loop over rows of irrep 
        for row in irrep:
#          correlator_row = np.zeros((0, ) + data[0].shape, dtype=np.double)
          correlator_row = []
          qn_row = []
          # loop over entries in CG-coefficient: First two elements are the 
          # momenta, last element is the corresponding coefficient for a 
          # spin singlet
          for so_3mom in row:
            for si_3mom in row:
              # enforce chosen momenta to belong to [k1, k2]
              if not (((np.dot(so_3mom[0], so_3mom[0]), \
                        np.dot(so_3mom[1], so_3mom[1])) in \
                                        [gevp_row, tuple(reversed(gevp_row))]) \
                     and ((np.dot(si_3mom[0], si_3mom[0]), 
                           np.dot(si_3mom[1], si_3mom[1])) in \
                                        [gevp_col, tuple(reversed(gevp_col))])):
                continue
    
              subduced = np.zeros((1,) + data[0].shape)
      
              # loop over contracted operators
              for op, qn in enumerate(qn_data):
                if not ((np.array_equal(so_3mom[0], qn[0]) and \
                                              np.array_equal(so_3mom[1], qn[3])) \
                        and (np.array_equal((-1)*si_3mom[0], qn[6]) and \
                                         np.array_equal((-1)*si_3mom[1], qn[9]))):
                  continue
     
                # CG = CG(source) * CG(sink)*
                factor = so_3mom[-1] * np.conj(si_3mom[-1])
                if factor == 0:
                  continue
#                subduced[0] = subduced[0] + factor.real * data[op].real + \
#                                                       factor.imag * data[op].imag
                subduced[0] = subduced[0] + (factor*data[op]).real

              # Omit correlator if no contracted operators are contributing
              if(subduced.any() != 0):
                correlator_row.append(np.squeeze(subduced, axis=0))
#                correlator_row = np.vstack((correlator_row, subduced))
                qn_row.append([ (so_3mom[0], so_3mom[1]), (si_3mom[0], si_3mom[1]), \
                     gevp_row, ('g5', 'g5'), gevp_col, ('g5', 'g5'), irreps[-1][i] ])
          # end loop over entries in CG-coefficient
          correlator_row = np.asarray(correlator_row)
          correlator_gevp_col.append(np.asarray(correlator_row))
          qn_gevp_col.append(np.asarray(qn_row))
        # end loop over rows of irrep 
        correlator_gevp_col = np.asarray(correlator_gevp_col)
        if(np.any(correlator_gevp_col != 0) and correlator_gevp_col.size != 0):
          correlator_gevp_row.append(correlator_gevp_col)
          qn_gevp_row.append(np.asarray(qn_gevp_col))
      correlator_gevp_row = np.asarray(correlator_gevp_row)
      if(np.any(correlator_gevp_row != 0) and correlator_gevp_row.size != 0):
        correlator_irrep.append(correlator_gevp_row)
        qn_irrep.append(np.asarray(qn_gevp_row))
    # end loop over [k1, k2]
    correlator.append(np.asarray(correlator_irrep))
    qn_subduced.append(np.asarray(qn_irrep))
  # end loop over irreps
  
  correlator = np.asarray(correlator)
  qn_subduced = np.asarray(qn_subduced)
  
  print correlator.shape
  print qn_subduced.shape
  
  #for i in correlator:
  #  for j in i:
  #    for k in j:
  #      print k.shape
  
  utils.ensure_dir('./readdata')
  utils.ensure_dir('./readdata/p%1i/' % p_cm)
  
  ################################################################################
  # write data to disc
  
  # write all subduced correlators
  path = './readdata/p%1i/%s_p%1i_subduced' % (p_cm, diagram, p_cm)
  np.save(path, correlator)
  path = './readdata/p%1i/%s_p%1i_subduced_qn' % (p_cm, diagram, p_cm)
  np.save(path, qn_subduced)
   
  # write means over all operators subducing into same irrep row
  print '\taveraging over momenta'
  path = './readdata/p%1i/%s_p%1i_subduced_avg_vecks' % (p_cm, diagram, p_cm)
  if correlator.ndim >= 5:
    avg = np.mean(correlator, axis=4)
  else:
    avg = []
    for i, irrep in enumerate(correlator):
      avg_irrep = []
      for k1k2, gevp_row in enumerate(irrep):
        avg_gevp_row = []
        for g2, gevp_col in enumerate(gevp_row):
          avg_gevp_col = []
          for r, row in enumerate(gevp_col):
            avg_gevp_col.append(np.sum(row, axis=0))
          avg_gevp_col = np.asarray(avg_gevp_col)
          avg_gevp_row.append(avg_gevp_col)
        avg_gevp_row = np.asarray(avg_gevp_row)
        avg_irrep.append(avg_gevp_row)
      avg_irrep = np.asarray(avg_irrep)
      avg.append(avg_irrep)
    avg = np.asarray(avg)
  print avg.shape
  np.save(path, avg)

  path = './readdata/p%1i/%s_p%1i_subduced_avg_vecks_qn' % (p_cm, diagram, p_cm)
  qn_avg = []
  for i, qn_irrep in enumerate(qn_subduced):
    qn_avg_irrep = []
    for k1, qn_gevp_row in enumerate(qn_irrep):
      qn_avg_gevp_row = []
      for k2, qn_gevp_col in enumerate(qn_gevp_row):
        qn_avg_gevp_col = []
        for r, qn_row in enumerate(qn_gevp_col):
          #inserting not necessary as gevp elements belong to qn
          qn_avg_gevp_col.append(np.asarray(qn_row[0][-5:]))
        qn_avg_gevp_col = np.asarray(qn_avg_gevp_col)
        qn_avg_gevp_row.append(qn_avg_gevp_col)
      qn_avg_gevp_row = np.asarray(qn_avg_gevp_row)
      qn_avg_irrep.append(qn_avg_gevp_row)
    qn_avg_irrep = np.asarray(qn_avg_irrep)
    qn_avg.append(qn_avg_irrep)
  qn_avg = np.asarray(qn_avg)
  np.save(path, qn_avg)

  np.save(path, qn_avg)

#  avg = np.zeros_like(correlator)
#  qn_avg = np.zeros_like(qn_subduced)
#  for i in range(correlator.shape[0]):
#    for k1 in range(correlator.shape[1]):
#      for k2 in range(correlator.shape[2]):
#        for r in range(correlator.shape[3]):
#          avg[i,k1,k2,r] = np.sum(correlator[i,k1,k2,r], axis=0) 
#          qn_avg[i,k1,k2,r] = qn_subduced[i,k1,k2,r][0,4:]
#  avg = np.asarray(avg.tolist())
#  print avg.shape
#  qn_avg = np.asarray(qn_avg.tolist())
  
  #write means over all operators subducing into same irrep
  path = './readdata/p%1i/%s_p%1i_subduced_avg_rows' % (p_cm, diagram, p_cm)
  print '\taveraging over rows'
  if avg.ndim >= 4:
    avg = np.mean(avg, axis=3)
  else:
    avg_tmp = []
    for i, irrep in enumerate(avg):
      avg_irrep = []
      for g1, gevp_row in enumerate(irrep):
        avg_gevp_row = []
        for g2, gevp_col in enumerate(gevp_row):
          avg_gevp_row.append(np.mean(gevp_col, axis=0))
        avg_gevp_row = np.asarray(avg_gevp_row)
        avg_irrep.append(avg_gevp_row)
      avg_irrep = np.asarray(avg_irrep)
      avg_tmp.append(avg_irrep)
    avg = np.asarray(avg_tmp)
  for c in avg:
    print c.shape
  np.save(path, avg)

  path = './readdata/p%1i/%s_p%1i_subduced_avg_rows_qn' % (p_cm, diagram, p_cm)
  qn_avg_tmp = []
  for i, qn_irrep in enumerate(qn_avg):
    qn_avg_irrep = []
    for g1, qn_gevp_row in enumerate(qn_irrep):
      qn_avg_gevp_row = []
      for g2, qn_gevp_col in enumerate(qn_gevp_row):
        qn_avg_gevp_row.append(qn_gevp_col[0])
      qn_avg_gevp_row = np.asarray(qn_avg_gevp_row)
      qn_avg_irrep.append(qn_avg_gevp_row)
    qn_avg_irrep = np.asarray(qn_avg_irrep)
    qn_avg_tmp.append(qn_avg_irrep)
  qn_avg = np.asarray(qn_avg_tmp)
  np.save(path, qn_avg)

