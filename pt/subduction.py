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

p_max = 4

verbose = 0

diagram = 'C2'

# Operators entering the Gevp. Last entry must contain the name in LaTeX 
# compatible notation for plot labels
gamma_i =   [1, 2, 3, 'gi']
gamma_0i =  [10, 11, 12, 'g0gi']
gamma_50i = [13, 14, 15, 'g5g0gi']
gamma_5 = [5, 'g5']

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
  
def set_lookup_p(p_max, p_cm, diagram):
  lookup_p3 = it.ifilter(lambda x: abs2(x) <= p_max, \
                         it.product(range(-p_max, p_max+1), repeat=3))
  
  if diagram == 'C3':
    lookup_p = it.ifilterfalse(lambda x: x[0] < x[1], \
        unique_everseen(it.starmap(lambda x, y: (abs2(x), abs2(y)), \
        it.ifilter(lambda (x,y): abs2(it.imap(operator.add, x, y)) == p_cm \
                   # for zero center of mass momentum omit the case were both 
                   # particles are at rest (s-wave)
                   and not (p_cm == 0 and tuple(x) == tuple(y) ) \
        ,it.product(lookup_p3, repeat=2)) \
          )) )
    #TODO: for p != 0 there are equivalent combinations with p1 <-> p2
    #TODO: critical for moving frames only group by summed 3-momentum at source and sink
    #for d in lookup_p:
    #  print d
  else:
    print 'in set_lookup_p: diagram unknown! Quantum numbers corrupted.'

  return list(lookup_p)

def get_irreps(p_cm, diagram, irrep):

  if diagram == 'C2':
    irreps_2pt = cg_2pt.coefficients(irrep)
    return irreps_2pt, irreps_2pt
  elif diagram == 'C3':
    # get factors for the desired irreps
    irreps_2pt = cg_2pt.coefficients(irrep)
    irreps_4pt = cg_4pt.coefficients(irrep)
    if len(irreps_4pt) != len(irreps_2pt):
      print 'in get_irreps: irrep for 2pt and 4pt functions contain ' \
            'different number of rows'
    # for 3pt function we have pipi operator at source and rho operator at sink
    return irreps_4pt, irreps_2pt
  else:
    print 'in get_irreps: diagram unknown! Quantum numbers corrupted.'
    return

def set_gevp_row(p_max, p_cm, gammas, diagram):

  if diagram == 'C2':
    return it.product([(p_cm,)], gammas)
  elif diagram == 'C3':
    lookup_p = set_lookup_p(p_max, p_cm, diagram)
    return it.product(lookup_p, [gamma_5])
  else:
    print 'in set_gevp_row: diagram unknown! Quantum numbers corrupted.'
    return

def set_gevp_col(p_max, p_cm, gammas, diagram):

  if diagram == 'C2':
    return it.product([(p_cm,)], gammas)
  elif diagram == 'C3':
    return it.product([(p_cm,)], gammas)
  else:
    print 'in set_gevp_col: diagram unknown! Quantum numbers corrupted.'
    return

def check_mom(so_3mom, si_3mom, gevp_row, gevp_col):

  if diagram == 'C2':
    if not (((np.dot(so_3mom[0], so_3mom[0]),) == gevp_row)
            and ((np.dot(si_3mom[0], si_3mom[0]),) == gevp_col)
            and np.array_equal(so_3mom[0], si_3mom[0])):
      return False
  elif diagram == 'C3':
    if not ((((np.dot(so_3mom[0], so_3mom[0]), \
                            np.dot(so_3mom[1], so_3mom[1])) == gevp_row) \
            or ((np.dot(so_3mom[1], so_3mom[1]), \
                            np.dot(so_3mom[0], so_3mom[0])) == gevp_row)) \
            and ((np.dot(si_3mom[0], si_3mom[0]),) == gevp_col)
            and np.array_equal(so_3mom[0]+so_3mom[1],si_3mom[0])):
      return False
  else:
    print 'in check_mom: diagram unknown! Quantum numbers corrupted.'
    return False

  return True

def check_qn(so_3mom, si_3mom, so_gamma, si_gamma, qn, diagram):
  if diagram == 'C2':
    if not (np.array_equal(so_3mom[0], qn[0]) \
       and np.array_equal((-1)*si_3mom[0], qn[3])):
      return False 
    if not ((so_gamma == qn[2]) and (si_gamma == qn[5])):
      return False
  elif diagram == 'C3':
    if not ((np.array_equal(so_3mom[0], qn[0]) and \
                              np.array_equal(so_3mom[1], qn[6])) \
       and (np.array_equal((-1)*si_3mom[0], qn[3]))):
      return False 
    if not ((so_gamma == qn[2]) and (si_gamma == qn[5])):
      return False
  else:
    print 'in check_qn: diagram unknown! Quantum numbers corrupted.'
    return False

  return True

def set_qn(so_3mom, si_3mom, gevp_row, p_cm, gevp_col, irrep):
  if diagram == 'C2':
    return [ so_3mom[0], (-1)*si_3mom[0], np.dot(so_3mom[0], so_3mom[0]), \
                              gevp_row[-1], np.dot(si_3mom[0], si_3mom[0]), \
                              gevp_col[-1], irrep ]
  elif diagram == 'C3':
    return [ (so_3mom[0], so_3mom[1]), si_3mom[0], gevp_row, ('g5', 'g5'), \
                                                     p_cm, gevp_col[-1], irrep ]
  else:
    print 'in set_qn: diagram unknown! Quantum numbers corrupted.'
    return


def ensembles(p_cm, diagram, p_max, gammas, verbose):

  print 'subducing p = %i' % p_cm

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
    irreps = ['T1']
  elif p_cm in [1,3,4]:
    irreps = ['A1', 'E2']
  elif p_cm in [2]:
    irreps = ['A1', 'B1', 'B2']
  else:
    # nothing to do here
    irreps = []
 

  correlator = []
  qn_subduced = []
  for irrep in irreps:
    irrep_so, irrep_si = get_irreps(p_cm, diagram, irrep)
    print 'subducing Lambda = %s' % irrep
    correlator_irrep = []
    qn_irrep = []
    for gevp_row in set_gevp_row(p_max, p_cm, gammas, diagram):
      if verbose:
        print gevp_row[0]
      correlator_gevp_row = []
      qn_gevp_row = []
      for gevp_col in set_gevp_col(p_max, p_cm, gammas, diagram):
        correlator_gevp_col = []
        qn_gevp_col = []
        for row in range(len(irrep_so)):
      
          correlator_row = []
          qn_row = []
      
          # only take coeffcients from momenta with the correct p_cm
          # alternatively: have all 3mom combinations in gevp_row (lookup_p)
          # alternatively: include p_cm into cg-coefficients
          # !!!!!
          for so_3mom in irrep_so[row]:
            for si_3mom in irrep_si[row]:
              if check_mom(so_3mom[0], si_3mom[0], gevp_row[0], gevp_col[0]):
#                if not (((np.dot(so_3mom[0][0], so_3mom[0][0]), \
#                                        np.dot(so_3mom[0][1], so_3mom[0][1])) == gevp_row[0]) \
#                        or ((np.dot(so_3mom[0][1], so_3mom[0][1]), \
#                                        np.dot(so_3mom[0][0], so_3mom[0][0])) == gevp_row[0])) \
#                        and ((np.dot(si_3mom[0][0], si_3mom[0][0]),) == gevp_col[0]):
#                  continue
      
                # in subduced all contributing qn (e.g. g1, g2, g3) are added up
                subduced = np.zeros((1,) + data[0].shape)
      
                # loop over all quantum numbers and check whether the current one
                # is the one wanted. With pandas just take the wanted ones.
                for op, qn in enumerate(qn_data):
                  for g_so in range(0,len(gevp_row[1])-1):
                    for g_si in range(0,len(gevp_col[1])-1):

                      # less efficient than checking after each loop but easier to generalize
                      # WARNING: we really loose a factor 3 here
                      if check_qn(so_3mom[0], si_3mom[0], gevp_row[1][g_so], \
                                                gevp_col[1][g_si], qn, diagram):

                        # !!!!
                        cg_factor = np.conj(so_3mom[g_so+1]) * si_3mom[g_si+1]
                        if cg_factor == 0:
                          continue
      
                        subduced[0] = subduced[0] + (cg_factor*data[op]).real
  
#                        print 'g_so = %i\t g_si = %i' % (gevp_row[1][g_so], gevp_col[1][g_si])
#                        print 'so_mom = (%i,%i,%i)\t si_mom = (%i,%i,%i)' % (so_3mom[0][0][0], so_3mom[0][0][1], so_3mom[0][0][2], si_3mom[0][0][0], si_3mom[0][0][1], si_3mom[0][0][2])
##                        if verbose:
#                          print '\tsubduced g_so = %i' % (gevp_col[1][g_si])
#                          print '\t\tsubduction coefficient = % .2f + % .2fi' % \
#                                                (cg_factor.real, cg_factor.imag)
#                          for j in range(3):
#                            print '\t\t\t', data[op][j][0]
#                          print ' '
#                          for j in range(3):
#                            print '\t\t\t', cg_factor*data[op][j][0]
    
                if(subduced.any() != 0):
#                  if verbose:
#                    print '\tinto momenta [(%i,%i,%i), (%i,%i,%i)]' % \
#                           (so_3mom[0][0][0], so_3mom[0][0][1], so_3mom[0][0][2], \
#                                        so_3mom[0][1][0], so_3mom[0][1][1], so_3mom[0][1][2])
                  correlator_row.append(np.squeeze(subduced, axis=0))
                  qn_row.append(set_qn(so_3mom[0], si_3mom[0], gevp_row[0], p_cm, gevp_col[1], irrep))
                  
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
      if(correlator_gevp_row.size != 0 and not 
          np.array_equal(correlator_gevp_row, np.zeros_like(correlator_gevp_row.shape))):
  #    if(np.any(correlator_gevp_row != 0) and correlator_gevp_row.size != 0):
        correlator_irrep.append(np.asarray(correlator_gevp_row))
        qn_irrep.append(np.asarray(qn_gevp_row))
#    print len(correlator_irrep)
#    tmp = np.array(correlator_irrep)
#    print 'shape: ', tmp.shape
    correlator.append(correlator_irrep)
    qn_subduced.append(qn_irrep)
  
  correlator = np.asarray(correlator)
  qn_subduced = np.asarray(qn_subduced)
  
#  print correlator.shape
#  for c in correlator:
#    print c.shape
  
  utils.ensure_dir('./readdata')
  utils.ensure_dir('./readdata/p%1i/' % p_cm)
  
  ################################################################################
  # write data to disc
  
  # write all subduced correlators
  path = './readdata/p%1i/%s_p%1i_subduced' % (p_cm, diagram, p_cm)
  np.save(path, correlator)
  path = './readdata/p%1i/%s_p%1i_subduced_qn' % (p_cm, diagram, p_cm)
  np.save(path, qn_subduced)
   
  # write all subduced correlators
  
  # write means over all three-vectors of operators subducing into same irrep, 
  # [k1,k2]-Gamma, mu
  print '\taveraging over momenta'
  path = './readdata/p%1i/%s_p%1i_subduced_avg_vecks' % (p_cm, diagram, p_cm)
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

  path = './readdata/p%1i/%s_p%1i_subduced_avg_vecks_qn' % (p_cm, diagram, p_cm)
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
  path = './readdata/p%1i/%s_p%1i_subduced_avg_rows' % (p_cm, diagram, p_cm)
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

  path = './readdata/p%1i/%s_p%1i_subduced_avg_rows_qn' % (p_cm, diagram, p_cm)
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
  

  
for p_cm in range(2):
  ensembles(p_cm, diagram, p_max, gammas, verbose)
  
