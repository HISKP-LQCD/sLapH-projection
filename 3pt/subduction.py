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

verbose = 1

diagram = 'C3+'

# Operators entering the Gevp. Last entry must contain the name in LaTeX 
# compatible notation for plot labels
gamma_i =   [1, 2, 3, 'gi']
gamma_0i =  [10, 11, 12, 'g0gi']
gamma_50i = [13, 14, 15, 'g5g0gi']

gammas = [gamma_i, gamma_0i, gamma_50i]


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
  
for p in range(0,5):
  print 'subducing p = %i' % p

  ################################################################################
  # read original data 
  
  path = './readdata/p%1i/%s_p%1i.npy' % (p, diagram, p)
  data = np.load(path)
  path = './readdata/p%1i/%s_p%1i_qn.npy' % (p, diagram, p)
  qn_data = np.load(path)
  if ( (qn_data.shape[0] != data.shape[0])):
    print '\tRead operators do not aggree with expected operators'
    exit(0)

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
  elif p in [1,3,4]:
    irreps_2pt = [['A1', 'E2']]
    irreps_4pt = [['A1', 'E2']]
  elif p in [2]:
    irreps_2pt = [['A1', 'B1', 'B2']]
    irreps_4pt = [['A1', 'B1', 'B2']]
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
    print 'subducing Lambda = %s' % irreps_2pt[-1][i]
    correlator_irrep = []
    qn_irrep = []
    for gevp_row in lookup_p:
      print gevp_row
      correlator_gevp_row = []
      qn_gevp_row = []
      for gevp_col in gammas:
        correlator_gevp_col = []
        qn_gevp_col = []
        for row in range(len(irrep_so)):
      
          correlator_row = []
          qn_row = []
      
          for so_3mom in irrep_so[row]:
            for si_3mom in irrep_si[row]:
              if not ((((np.dot(so_3mom[0], so_3mom[0]), \
                                      np.dot(so_3mom[1], so_3mom[1])) == gevp_row) \
                      or ((np.dot(so_3mom[0], so_3mom[0]), \
                                      np.dot(so_3mom[1], so_3mom[1])) == tuple(reversed(gevp_row)))) \
                     and (np.dot(si_3mom[0], si_3mom[0]) == p)):
                continue
      
              # in subduced all contributing qn (e.g. g1, g2, g3) are added up
              subduced = np.zeros((1,) + data[0].shape)
      
              for op, qn in enumerate(qn_data):
                if not ((np.array_equal(so_3mom[0], qn[0]) and \
                                              np.array_equal(so_3mom[1], qn[6])) \
                       and (np.array_equal((-1)*si_3mom[0], qn[3]))):
                  continue
                for g_si in range(0,3):
      
#                  cg_factor = so_3mom[-1] * np.conj(si_3mom[g_si+1])
                  cg_factor = np.conj(so_3mom[-1]) * si_3mom[g_si+1]
#                  cg_factor = so_3mom[-1] * si_3mom[g_si+1]
                  if cg_factor == 0:
                    continue

                  if gevp_col[g_si] in gamma_i:
                    wick_factor = 2.
                  elif gevp_col[g_si] in gamma_0i:
                    wick_factor = -2.
                  elif gevp_col[g_si] in gamma_50i:
                    wick_factor = 2*1j

                  factor = cg_factor*wick_factor
                  if (gevp_col[g_si] == qn[5]):
#                    subduced[0] = subduced[0] + (factor*data[op]).real
                    subduced[0] = subduced[0] + (factor*data[op]).real
                    if verbose:
                      print '\tsubduced g_so = %i' % (gevp_col[g_si])
                      print '\t\tsubduction coefficient = % .2f + % .2fi' % \
                                                        (factor.real, factor.imag)
                      for j in range(3):
                        print '\t\t\t', data[op][j][0]
                      print ' '
                      for j in range(3):
                        print '\t\t\t', factor*data[op][j][0]
  
              if(subduced.any() != 0):
                if verbose:
                  print '\tinto momenta [(%i,%i,%i), (%i,%i,%i)]' % \
                         (so_3mom[0][0], so_3mom[0][1], so_3mom[0][2], \
                                      so_3mom[1][0], so_3mom[1][1], so_3mom[1][2])
                correlator_row.append(np.squeeze(subduced, axis=0))
                qn_row.append([ (so_3mom[0], so_3mom[1]), si_3mom[0], \
                              gevp_row, ('g5', 'g5'), p, gevp_col[-1], irreps_4pt[-1][i] ])
                  
          if len(correlator_row) == 0: 
            continue
          correlator_row = np.asarray(correlator_row)
          correlator_gevp_col.append(correlator_row)
          qn_row = np.asarray(qn_row, dtype=object)
          qn_gevp_col.append(qn_row)
        if len(correlator_gevp_col) == 0:
          continue
        correlator_gevp_row.append(np.asarray(correlator_gevp_col))
        qn_gevp_row.append(np.asarray(qn_gevp_col))
      correlator_gevp_row = np.asarray(correlator_gevp_row)
      if(np.any(correlator_gevp_row != 0) and correlator_gevp_row.size != 0):
        correlator_irrep.append(np.asarray(correlator_gevp_row))
        qn_irrep.append(np.asarray(qn_gevp_row))
    correlator.append(np.asarray(correlator_irrep))
    qn_subduced.append(np.asarray(qn_irrep))
  
  correlator = np.asarray(correlator)
  qn_subduced = np.asarray(qn_subduced)
  
  print correlator.shape
  for c in correlator:
    print c.shape
  
  utils.ensure_dir('./readdata')
  utils.ensure_dir('./readdata/p%1i/' % p)
  
  ################################################################################
  # write data to disc
  
  # write all subduced correlators
  path = './readdata/p%1i/%s_p%1i_subduced' % (p, diagram, p)
  np.save(path, correlator)
  path = './readdata/p%1i/%s_p%1i_subduced_qn' % (p, diagram, p)
  np.save(path, qn_subduced)
   
  # write all subduced correlators
  
  # write means over all three-vectors of operators subducing into same irrep, 
  # [k1,k2]-Gamma, mu
  print '\taveraging over momenta'
  path = './readdata/p%1i/%s_p%1i_subduced_avg_vecks' % (p, diagram, p)
  if correlator.ndim >= 5:
    avg = np.sum(correlator, axis=4)
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

  path = './readdata/p%1i/%s_p%1i_subduced_avg_vecks_qn' % (p, diagram, p)
  qn_avg = []
  for i, qn_irrep in enumerate(qn_subduced):
    qn_avg_irrep = []
    for g1, qn_gevp_row in enumerate(qn_irrep):
      qn_avg_gevp_row = []
      for g2, qn_gevp_col in enumerate(qn_gevp_row):
        qn_avg_gevp_col = []
        for r, qn_row in enumerate(qn_gevp_col):
#          qn_avg_row = []
#          for k, qn_vec in enumerate(qn_row):
#            qn_avg_row.append(np.asarray([np.dot(qn_avg_vec[1], qn_avg_vec[1]), \
#                               np.dot(qn_avg_vec[1], qn_avg_vec[1]), \
#                                                              qn_avg_vec[-3:]]))
#            qn_avg_row.append(np.insert( \
#                np.insert( qn_vec[-5:], 0, \
#                    (np.dot(qn_vec[-7][1], qn_vec[-7][1]), np.dot(qn_vec[-7][0], qn_vec[-7][0])), \
#                    axis=-1), \
#                0, np.dot(qn_vec[-6], qn_vec[-6]), axis=-1))

          #inserting not necessary as gevp elements belong to qn
#          print r, qn_row.shape, qn_row[0][-5:]
#          qn_avg_row = qn_row[0][-5:]
#          qn_avg_row = np.asarray(qn_avg_row)
#          print qn_avg_row.shape
          qn_avg_gevp_col.append(qn_row[0][-5:])
        qn_avg_gevp_col = np.asarray(qn_avg_gevp_col)
        qn_avg_gevp_row.append(qn_avg_gevp_col)
      qn_avg_gevp_row = np.asarray(qn_avg_gevp_row)
      qn_avg_irrep.append(qn_avg_gevp_row)
    qn_avg_irrep = np.asarray(qn_avg_irrep)
    qn_avg.append(qn_avg_irrep)
  qn_avg = np.asarray(qn_avg)
  np.save(path, qn_avg)

#  avg = np.zeros_like(correlator)
#  qn_avg = np.zeros_like(qn_subduced)
#  for i in range(correlator.shape[0]):
#    for k in range(correlator.shape[1]):
#      for g in range(correlator.shape[2]):
#        for r in range(correlator.shape[3]):
#          avg[i,k,g,r] = np.sum(correlator[i,k,g,r], axis=0) 
#          qn_avg[i,k,g,r] = qn_subduced[i,k,g,r][0,3:]
#  avg = np.asarray(avg.tolist())
#  qn_avg = np.asarray(qn_avg.tolist())
  
  print '\taveraging over rows'
  # write means over all rows of operators subducing into same irrep, [k1,k2]
  path = './readdata/p%1i/%s_p%1i_subduced_avg_rows' % (p, diagram, p)
#  avg = np.mean(avg, axis=-3)
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
  print avg.shape
  np.save(path, avg)

  path = './readdata/p%1i/%s_p%1i_subduced_avg_rows_qn' % (p, diagram, p)
#  qn_avg = qn_avg[...,0,:]
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
  
  for c in avg:
    print c.shape
  
  
  
