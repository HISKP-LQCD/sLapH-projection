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

# TODO: should be a more elegant solution for get_gevp_gamma and 
def get_gevp_single_gamma(g):

  # Operators entering the Gevp. Last entry must contain the name in LaTeX 
  # compatible notation for plot labels
  gamma_i =   [1, 2, 3, 'gi']
  gamma_0i =  [10, 11, 12, 'g0gi']
  gamma_50i = [13, 14, 15, 'g5g0gi']
  gamma_5 = [5, 'g5']

  if g in gamma_i:
    return '\gamma_i'
  elif g in gamma_0i:
    return '\gamma_0i'
  elif g in gamma_50i:
    return '\gamma_50i'
  elif g in gamma_5:
    return '\gamma_5'

def get_gevp_gamma(gamma):
  if (len(gamma) == 1):
    return get_gevp_single_gamma(gamma[0])
  else:
    result = []
    for g in gamma:
      result.append(get_gevp_single_gamma(g))
    return tuple(result)


# TODO: change p_cm to string symmetry_group in get_clebsch_gordans
# TODO: find a better name for diagram after Wick contraction
def get_clebsch_gordans(diagram, p_cm, irrep):
  """
  Read table with required lattice Clebsch-Gordan coefficients

  Parameters
  ----------  
  diagram : string, {'C2', 'C3', 'C4'}
      Type of correlation function contributing to gevp. The number is the 
      number of quarklines
  p_cm : int
      Center of mass momentum of the lattice. Used to specify the appropriate
      little group of rotational symmetry
  irrep : string, {'T1', 'A1', 'E2', 'B1', 'B2'}
      name of the irreducible representation of the little group the operator
      is required to transform under.

  Returns
  ------- 
  cg_table : pd.DataFrame
      Table with all quantum numbers on source and sink as columns and the rows
      of the irreducible representations as rows. Also contains two columns
      with the appropriate Clebsch Gordan coefficient for all these quantum
      numbers

  Notes
  -----
  The Clebsch-Gordan coefficients are hardcoded and listed in two python files
  clebsch_gordan_2pt.py and clebsch_gordan_4pt.py. In the future they will be
  dynamically generated in the clebsch_gordan submodule
  """

  if diagram == 'C2':
    cg_one_operators = cg_2pt.coefficients(irrep)
    cg_table_so, cg_table_si = cg_one_operators, cg_one_operators
  elif diagram == 'C3':
    # get factors for the desired irreps
    cg_one_operators = cg_2pt.coefficients(irrep)
    cg_two_operators = cg_4pt.coefficients(irrep)
    if len(cg_two_operators) != len(cg_one_operators):
      print 'in get_clebsch_gordans: irrep for 2pt and 4pt functions contain ' \
            'different number of rows'
    # for 3pt function we have pipi operator at source and rho operator at sink
    cg_table_so, cg_table_si = cg_two_operators, cg_one_operators
  elif diagram == 'C4':
    cg_two_operators = cg_4pt.coefficients(irrep)
    cg_table_so, cg_table_si = cg_two_operators, cg_two_operators
  else:
    print 'in get_clebsch_gordans: diagram unknown! Quantum numbers corrupted.'
    return
  # combine clebsch-gordan coefficients for source and sink into one DataFrame
  cg_table = pd.merge(cg_table_so, cg_table_si, how='inner', \
      left_index=True, right_index=True, suffixes=['_{so}', '_{si}']) 
  return cg_table

def get_qn_irrep(qn, diagram, p_cm, irrep):
  """
  Read table with required lattice Clebsch-Gordan coefficients

  Parameters
  ----------  
  qn : pd.DataFrame
      pd.DataFrame with every row being a set of physical quantum numbers
      contributing to `diagram`. The rows are indexed by a unique identifier
  diagram : string, {'C2', 'C3', 'C4'}
      Type of correlation function contributing to gevp. The number is the 
      number of quarklines
  p_cm : int
      Center of mass momentum of the lattice. Used to specify the appropriate
      little group of rotational symmetry
  irrep : string, {'T1', 'A1', 'E2', 'B1', 'B2'}
      name of the irreducible representation of the little group the operator
      is required to transform under.

  Returns
  ------- 
  qn_irrep : pd.DataFrame
      Table with a column for each quantum number at source and sink, the
      Clebsch-Gordan coefficients at source and sink and a column decoding
      the row and column of the gevp these quantum numbers enter into:
      \mu \gamma_{so} cg-coefficient_{so} p_{so} \gamma_{si} \
          cg-coefficient_{si} p_{si} index gevp_row gevp_col
  """

  # read Clebsch-Gordan coefficients for correlation function, little group
  # and irreducible representation given by diagram, p_cm and irrep
  cg_table = get_clebsch_gordans(diagram, p_cm, irrep)

  print cg_table
  
  # associate clebsch-gordan coefficients with the correct qn index
  qn_irrep = pd.merge(cg_table.reset_index(), qn.reset_index())

  # Add two additional columns with the same string if the quantum numbers 
  # describe equivalent physical constellations
  # TODO: check whether that works for 4pt function
                              #apply(lambda x: tuple(list(x))).astype(tuple).astype(str) \
  qn_irrep['gevp_row'] = 'p = ' + \
                            qn_irrep['p_{so}'].apply(np.array).apply(np.square).\
                              apply(functools.partial(np.sum, axis=-1)).\
                              astype(tuple).astype(str) \
                         + ' \gamma = ' + \
                            qn_irrep['\gamma_{so}'].apply(get_gevp_gamma).astype(str) 
  qn_irrep['gevp_col'] = 'p = ' + \
                            qn_irrep['p_{si}'].apply(np.array).apply(np.square).\
                              apply(functools.partial(np.sum, axis=-1)).\
                              astype(tuple).astype(str) \
                          + ' \gamma = ' + \
                            qn_irrep['\gamma_{si}'].apply(get_gevp_gamma).astype(str)
  return qn_irrep


def ensembles(p_cm, diagram, p_max, verbose):

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

  irrep = 'T1'
  qn_irrep = get_qn_irrep(qn, diagram, p_cm, irrep)

  if verbose:
    print "qn_irrep:"
    print qn_irrep
  
  # actual subduction step. sum cg_so * conj(cg_si) * corr
  subduced = pd.merge(qn_irrep, data.T, how='left', left_on=['index'], 
                                                               right_index=True)
  # not needed after index was merged on
  del subduced['index']
  # construct hierarchical multiindex to be able to sum over momenta, average
  # over rows and reference gevp elements
  subduced = subduced.set_index(['gevp_row', 'gevp_col', '\mu', \
                      'p_{so}', '\gamma_{so}', 'p_{si}', '\gamma_{si}'])
  subduced = subduced.ix[:,2:].multiply(subduced['cg-coefficient_{so}']*
                               np.conj(subduced['cg-coefficient_{si}']), axis=0)
  subduced.columns=pd.MultiIndex.from_tuples(subduced.columns)
  subduced.sort_index()

  # create full correlators as real + 1j * imag and save real and imaginary 
  # part seperately
  subduced = subduced.xs('re', level=2, axis=1) + \
                                       (1j) * subduced.xs('im', level=2, axis=1)

  # sum over gamma structures. 
  # Only real part is physically relevant at that point
  subduced = subduced.apply(np.real).sum(level=[0,1,2,3,5])
  # sum over equivalent momenta
  subduced_sum_mom = subduced.sum(level=[0,1,2])
  # average over rows
  subduced_sum_mom_avg_row = subduced_sum_mom.mean(level=[0,1])

  ##############################################################################
  # write data to disc

  utils.ensure_dir('./readdata')
  utils.ensure_dir('./readdata/p%1i/' % p_cm)

  store = pd.HDFStore('./readdata/%s_p%1i_subduced.h5' % (diagram, p_cm))
  # write all operators
  store['data'] = subduced_sum_mom_avg_row
  store['single correlators'] = subduced

  store.close()
  
  print '\tfinished writing'
 
  return subduced_sum_mom_avg_row
#for p_cm in range(2):
#  ensembles(p_cm, diagram, p_max, verbose)
