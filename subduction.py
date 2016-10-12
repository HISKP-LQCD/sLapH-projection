#!/usr/bin/python

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import itertools as it
import cmath
import functools

import operator
import collections

import clebsch_gordan_2pt as cg_2pt
import clebsch_gordan_4pt as cg_4pt
import utils


p_max = 4

verbose = 0

diagram = 'C3'

# Operators entering the Gevp. Last entry must contain the name in LaTeX 
# compatible notation for plot labels
gamma_i =   [1, 2, 3, 'gi']
gamma_0i =  [10, 11, 12, 'g0gi']
gamma_50i = [13, 14, 15, 'g5g0gi']
gamma_5 = [5, 'g5']

gammas = [gamma_i, gamma_0i, gamma_50i]

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
  elif diagram == 'C4':
    irreps_4pt = cg_4pt.coefficients(irrep)
    return irreps_4pt, irreps_4pt
  else:
    print 'in get_irreps: diagram unknown! Quantum numbers corrupted.'
    return

# TODO: should be a more elegant solution for get_gevp_gamma and 
def get_gevp_single_gamma(g):
  if g in gamma_i:
    return '\gamma_i'
  elif g in gamma_0i:
    return '\gamma_0i'
  elif g in gamma_50i:
    return '\gamma_50i'
  elif g in gamma_5:
    return '\gamma_5'

def get_gevp_gamma(gamma):
  result = []
  for g in gamma:
    result.append(get_gevp_single_gamma(g))
  return tuple(result)


def ensembles(p_cm, diagram, p_max, gammas, verbose):

  print 'subducing p = %i' % p_cm

  ################################################################################
  # read original data 
  data = pd.read_hdf('readdata/%s_p%1i.h5' % (diagram, p_cm), 'data')
  data.columns.name = 'index'
  qn = pd.read_hdf('readdata/%s_p%1i.h5' % (diagram, p_cm), 'qn')

  if p_cm in [0]:
    irreps = ['T1']
  elif p_cm in [1,3,4]:
    irreps = ['A1', 'E2']
  elif p_cm in [2]:
    irreps = ['A1', 'B1', 'B2']
  else:
    # nothing to do here
    irreps = []

  # combine clebsch-gordan coefficients for source and sink into one DataFrame
  irrep_so, irrep_si = get_irreps(0, 'C3', 'T1')
  irrep = pd.merge(irrep_so, irrep_si, how='inner', \
      left_index=True, right_index=True, suffixes=['_{so}', '_{si}']) 
  
  # associate clebsch-gordan coefficients with the correct correlators
  qn_irrep = pd.merge(irrep.reset_index(), qn.reset_index())

  # set entries for gevp
  qn_irrep['gevp_row'] = 'p = ' + \
                            qn_irrep['p_{so}'].apply(np.array).apply(np.square).\
                              apply(functools.partial(np.sum, axis=-1)).\
                              apply(lambda x: tuple(x)).astype(tuple).astype(str) \
                         + ' \gamma = ' + \
                            qn_irrep['\gamma_{so}'].apply(get_gevp_gamma).astype(str) 
  qn_irrep['gevp_col'] = 'p = ' + \
                            qn_irrep['p_{si}'].apply(np.array).apply(np.square).\
                              apply(functools.partial(np.sum, axis=-1)).\
                              astype(tuple).astype(str) \
                          + ' \gamma = ' + \
                            qn_irrep['\gamma_{si}'].apply(get_gevp_gamma).astype(str)
  
  # actual subduction step. sum cg_so * conj(cg_si) * corr
  subduced = pd.merge(qn_irrep, data.T, how='left', left_on=['index'], right_index=True)
  del subduced['index']
  subduced = subduced.set_index(['gevp_row', 'gevp_col', '\mu', \
                      'p_{so}', '\gamma_{so}', 'p_{si}', '\gamma_{si}'])
  subduced = subduced.ix[:,2:].multiply(subduced['cg-coefficient_{so}']*np.conj(subduced['cg-coefficient_{si}']), axis=0)
  subduced.columns=pd.MultiIndex.from_tuples(subduced.columns)
  subduced.sort_index()
  subduced = subduced.sum(level=[0,1,2,3,5])

  # create full correlators as real + 1j * imag
  # TODO: with proper slicing this is a oneliner
  subduced = subduced.swaplevel(0,2,axis=1)
  subduced = (subduced['re'] + (1j) * subduced['im']).apply(np.real)
  subduced = subduced.swaplevel(axis=1)

  return subduced

  # sum over equivalent momenta
  subduced_sum_mom = subduced.sum(level=[0,1,2])

  # average over rows
  subduced_sum_mom_avg_row = subuduced_sum_mom.mean(level=[0,1])

  return subduced_sum_mom_avg_row

  ################################################################################
  # write data to disc

  utils.ensure_dir('./readdata')
  utils.ensure_dir('./readdata/p%1i/' % p_cm)

  store = pd.HDFStore('./readdata/%s_p%1i_subduced.h5' % (diagram, p_cm))
  # write all operators
  store['data'] = subduced_sum_mom_avg_row
  store['single correlators'] = subduced

  store.close()
  
  print '\tfinished writing'
 
#for p_cm in range(2):
#  ensembles(p_cm, diagram, p_max, gammas, verbose)
