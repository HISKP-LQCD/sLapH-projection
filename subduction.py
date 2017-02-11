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


def set_lookup_irreps(p_cm):
  """
  Calculate list of irreducible representations contributing to the appropriate 
  little group of rotational symmetry

  Parameters
  ----------
    p_cm : int, {0,1,2,3,4}
        Center of mass momentum of the lattice. Used to specify the appropriate
        little group of rotational symmetry

  Returns
  -------
    irreps : list of strings
        List with the namees of all contributing irreducible representations
  """
  if p_cm in [0]:
    irreps = ['T1']
  elif p_cm in [1,3,4]:
    irreps = ['A1', 'E2']
  elif p_cm in [2]:
    irreps = ['A1', 'B1', 'B2']
  else:
    # nothing to do here
    irreps = []
  return irreps

# TODO: properly read that from infile and pass to get_clebsch_gordan
# TODO: actually use names to restrict basis_table to what was in the infile
def get_basis(names, verbose):
  """
  Get table with chosen operators to transform as the *continuum* eigenstates
  of the vector spin representation

  Parameters
  ----------
  names : list of string
      Contains the names of chosen multiplets as the set of continuum 
      eigenstates is ambiguous and multiple choices might be wanted for a gevp.

  Returns
  -------
  basis_table : pd.DataFrame
      Table with linear coefficients to contruct a spin 1 eigenstate from 
      operators. Rows are hierarchical containing the `name` of a multiplet and 
      the eigenstates it contains. There one column for each linearlz 
      independent Lorentz structure.
  """

#  # hardcode ladder operators J_+, J_3 and J_-
#  # Dudek helicity basis
#  sqrt2 = np.sqrt(2.)
#  ladder_operators = [[-1j/sqrt2, -1./sqrt2, 0], 
#                      [ 0,         0,        1j], 
#                      [ 1j/sqrt2, -1./sqrt2, 0]]
  # implement trivial orthogonal basis as it is taken account for in 
  # cg coefficients
  ladder_operators = [[1, 0, 0], 
                      [0, 1, 0], 
                      [0, 0, 1]]

  basis = np.array([m + [0]*3 for m in ladder_operators] + \
                                  [[0]*3+m for m in ladder_operators]).flatten()
  # hardcode basis operators for \gamma_i and \gamma_5\gamma_0\gamma_i
  # TODO: replace first list in MultiIndex.from_product by names (or some kind 
  # of latex(names)
  basis_table = DataFrame(basis, \
            index=pd.MultiIndex.from_product( \
                [["\gamma{_i}  ", "\gamma_{50i}"], \
                 ["|1,+1\rangle", "|1, 0\rangle", "|1,-1\rangle"], \
                                     [(1,), (2,), (3,), (13,), (14,), (15,)]], \
                names=["gevp", '|J, M\rangle', '\gamma']), \
            columns=['subduction-coefficient'], dtype=complex).sort_index()
  # conatenate basis chosen for two-pion operator. Trivial, because just a 
  # singlet state.
  basis_table = pd.concat([basis_table, DataFrame([1], \
            index=pd.MultiIndex.from_product( \
                [["(\gamma_{5}, \gamma_{5})"], ["|0, 0\rangle"], [(5,5)]], \
                names=["gevp", '|J, M\rangle', '\gamma']), \
            columns=['subduction-coefficient'], dtype=complex).sort_index()])

  return basis_table[basis_table['subduction-coefficient'] != 0]


# TODO: change p_cm to string symmetry_group in get_clebsch_gordans
# TODO: find a better name for diagram after Wick contraction
def get_clebsch_gordans(diagram, gammas, p_cm, irrep, verbose):
  """
  Read table with required lattice Clebsch-Gordan coefficients

  Parameters
  ----------  
  diagram : string, {'C2', 'C3', 'C4'}
      Type of correlation function contributing to gevp. The number is the 
      number of quarklines
  gammas : list of string
      Contains the names of chosen multiplets as the set of continuum 
      eigenstates is ambiguous and multiple choices might be wanted for a gevp.
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

  if diagram.startswith('C2'):
    cg_one_operator = cg_2pt.coefficients(p_cm, irrep)
    cg_table_so, cg_table_si = cg_one_operator, cg_one_operator
  elif diagram.startswith('C3'):
    # get factors for the desired irreps
    cg_one_operator = cg_2pt.coefficients(p_cm, irrep)
    cg_two_operators = cg_4pt.coefficients(irrep)
    # Chistopher Thomas: Written down for Creation operator
    cg_two_operators['cg-coefficient'] = np.conj(cg_two_operators['cg-coefficient'])
    # for 3pt function we have pipi operator at source and rho operator at sink
    cg_table_so, cg_table_si = cg_two_operators, cg_one_operator
  elif diagram.startswith('C4'):
    cg_two_operators = cg_4pt.coefficients(irrep)
    # Christopher Thomas: Written down for Creation operator
    cg_two_operators['cg-coefficient'] = np.conj(cg_two_operators['cg-coefficient'])
    cg_table_so, cg_table_si = cg_two_operators, cg_two_operators
  else:
    print 'in get_clebsch_gordans: diagram unknown! Quantum numbers corrupted.'
    return

  # express basis states for all Lorentz structures in `gammas` in terms of 
  # physical Dirac operators
  basis_table = get_basis(gammas, verbose)

  # express the subduced eigenstates in terms of Dirac operators.
  cg_table_so = pd.merge(cg_table_so.reset_index(), basis_table.reset_index()).\
                                                   set_index('\mu').sort_index()   
  cg_table_si = pd.merge(cg_table_si.reset_index(), basis_table.reset_index()).\
                                                   set_index('\mu').sort_index()  
  # Munging the result: Delete rows with coefficient 0, combine coefficients 
  # and clean columns no longer needed.
  cg_table_so = cg_table_so[cg_table_so['cg-coefficient'] != 0]
  cg_table_so['coefficient'] = \
           cg_table_so['cg-coefficient'] * cg_table_so['subduction-coefficient']
  cg_table_so.drop(['|J, M\rangle', 'cg-coefficient', 'subduction-coefficient'], \
                                                           axis=1, inplace=True)
  cg_table_si = cg_table_si[cg_table_si['cg-coefficient'] != 0]
  cg_table_si['coefficient'] = \
           cg_table_si['cg-coefficient'] * cg_table_si['subduction-coefficient']

  cg_table_si.drop(['|J, M\rangle', 'cg-coefficient', 'subduction-coefficient'], \
                                                           axis=1, inplace=True)

  # combine clebsch-gordan coefficients for source and sink into one DataFrame
  cg_table = pd.merge(cg_table_so, cg_table_si, how='inner', \
      left_index=True, right_index=True, suffixes=['_{so}', '_{si}']) 
  return cg_table

def set_lookup_qn_irrep(qn, diagram, gammas, p_cm, irrep, verbose):
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
  cg_table = get_clebsch_gordans(diagram, gammas, p_cm, irrep, verbose)

  # associate clebsch-gordan coefficients with the correct qn index
  qn_irrep = pd.merge(cg_table.reset_index(), qn.reset_index())

  # Add two additional columns with the same string if the quantum numbers 
  # describe equivalent physical constellations: gevp_row and gevp_col
  qn_irrep['gevp_row'] = 'p = ' + \
                            qn_irrep['p_{so}'].apply(np.array).apply(np.square).\
                              apply(functools.partial(np.sum, axis=-1)).\
                              astype(tuple).astype(str) \
                         + ', \gamma = ' + \
                            qn_irrep['gevp_{so}']
  qn_irrep['gevp_col'] = 'p = ' + \
                            qn_irrep['p_{si}'].apply(np.array).apply(np.square).\
                              apply(functools.partial(np.sum, axis=-1)).\
                              astype(tuple).astype(str) \
                          + ', \gamma = ' + \
                            qn_irrep['gevp_{si}']
  del(qn_irrep['gevp_{so}'])
  del(qn_irrep['gevp_{si}'])

  return qn_irrep


def ensembles(data, qn_irrep):
  """
  Combine physical operators to transform like a given irreducible 
  representation and combine equivalent physical quantum numbers

  Parameters
  ----------
  data : pd.DataFrame
      Cleaned und munged raw output of the cntr-v.0.1 code
  qn_irrep : pd.Series
      Series with a column for each quantum number at source and sink, the
      Clebsch-Gordan coefficients at source and sink and a column decoding
      the row and column of the gevp these quantum numbers enter into:
      \mu \gamma_{so} cg-coefficient_{so} p_{so} \gamma_{si} \
          cg-coefficient_{si} p_{si} index gevp_row gevp_col

  Returns
  -------
  subduced : pd.DataFrame
      Table with physical quantum numbers as rows and gauge configuration and 
      lattice time as columns.
      Contains the linear combinations of correlation functions transforming
      like what the parameters qn_irrep was created with
  """

  # actual subduction step. sum cg_so * conj(cg_si) * corr
  # TODO: This generates a warning 
  # /hadron/werner/.local/lib/python2.7/site-packages/pandas/tools/merge.py:480: UserWarning: merging between different levels can give an unintended result (1 levels on the left, 2 on the right)
  #  warnings.warn(msg, UserWarning)
  # But the merging is on one level only.
  subduced = pd.merge(qn_irrep, data.T, how='left', left_on=['index'], 
                                                               right_index=True)
  # not needed after index was merged on
  del subduced['index']
  # construct hierarchical multiindex to be able to sum over momenta, average
  # over rows and reference gevp elements
  subduced = subduced.set_index(['gevp_row', 'gevp_col', '\mu', \
                      'p_{so}', '\gamma_{so}', 'p_{si}', '\gamma_{si}'])
  subduced = subduced.ix[:,2:].multiply(subduced['coefficient_{so}']*
                               np.conj(subduced['coefficient_{si}']), axis=0)
  subduced.columns=pd.MultiIndex.from_tuples(subduced.columns, \
                                                         names=('cnfg', 'T'))

  return subduced.sort_index()

