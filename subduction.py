#!/usr/bin/python

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import itertools as it
import cmath
import functools
import os

import operator
import collections

from asteval import Interpreter
aeval = Interpreter()
aeval.symtable['I'] = 1j

#import clebsch_gordan_2pt as cg_2pt
#import clebsch_gordan_4pt as cg_4pt
import utils

from clebsch_gordan import group

def select_irrep(df, irrep):
  """
  Restrict table of basis vectors to one irreducible representation.

  Parameters
  ----------
    df : pd.DataFrame
        Contains subduction coefficients for going from continuum to discete
        space. 
    irrep : string
        Specifying the irreducible representation operators should transform 
        under

  Returns
  -------
        df restricted to *irrep* and *mult*. 
        Has columns J, M, cg-coefficient, p, \mu and unnamed indices

  See
  ---
    get_lattice_basis()
  """

  df = df[df['Irrep'] == irrep]
  df.drop(['Irrep'], axis=1, inplace=True)

  return df


def select_irrep_mult(df, irrep, mult):
  """
  Restrict table of basis vectors to one irreducible representation.

  Parameters
  ----------
    df : pd.DataFrame
        Contains subduction coefficients for going from continuum to discete
        space. 
    irrep : string
        Specifying the irreducible representation operators should transform 
        under
    mult : int
        The multiplicity further specifying the irreducible representation if
        it appears multiple times in the subduction process

  Returns
  -------
        df restricted to *irrep* and *mult*. 
        Has columns J, M, cg-coefficient, p, \mu and unnamed indices

  See
  ---
    get_lattice_basis()
  """

  df = df[df['Irrep'] == irrep]
  df = df[df['mult'] == mult]
  df.drop(['Irrep', 'mult'], axis=1, inplace=True)

  return df

# TODO: path for groups is hardcoded here. Shift that into clebsch-gordan module
def return_cg(p_cm, irrep):
  """
  Creates table with eigenstates of an irreducible representation created from
  a Clebsch-Gordan decomposition of two pseudoscalar particles with momenta 
  k_1 and k_2

  Parameters
  ----------

    p_cm : int, {0,1,2,3,4}
        Center of mass momentum of the lattice. Used to specify the appropriate
        little group of rotational symmetry. Absolute value of an integer 
        3-vector
    irrep : string
        Specifying the irreducible representation operators should transform 
        under

  Returns
  -------

    pd.DataFrame
        Table with cg-coefficients for each momentum combination p=(k_1, k_2)
        and row \mu
        Has columns J, M, cg-coefficient, p, \mu and unnamed indices

  Notes
  -----

    J, M are both hardcoded to (0,0) referring to scattering of two 
    (pseudo)scalars

  See
  ---

    clebsch_gordan.example_cg
  """


  prefs = [[0.,0.,0.], [0.,0.,1.], [0.,1.,1.], [1.,1.,1.], [0.,0.,2.]]
#           [0.,1.,2.], [1.,1.,2.]]
  p2max = len(prefs)

  # initialize groups
  S = 1./np.sqrt(2.)
  # tells clebsch_gordan to use cartesian basis
  U3 = np.asarray([[0,0,-1.],[1.j,0,0],[0,1,0]])
  U2 = np.asarray([[S,S],[1.j*S,-1.j*S]])

  path = os.path.normpath(os.path.join(os.getcwd(), "groups/"))
  groups = group.init_groups(prefs=prefs, p2max=p2max, U2=U2, U3=U3,
          path=path)

  # define the particles to combine
  j1 = 0 # J quantum number of particle 1
  j2 = 0 # J quantum number of particle 2
  ir1 = [ g.subduction_SU2(int(j1*2+1)) for g in groups]
  ir2 = [ g.subduction_SU2(int(j2*2+1)) for g in groups]

  # calc coefficients
  df = DataFrame()
  for (i, i1), (j, i2) in it.product(zip(range(p2max), ir1), zip(range(p2max), ir2)):
    for _i1, _i2 in it.product(i1, i2):
      try:

#        #hardcoded from cntr.v0.1: Cutoffs for single momenta
        # not really needed. Makes cg-calculation more efficient, but merging 
        # takes care of superfluous coefficients anyway
#        if p_cm == 0:
#          if p > 3:
#            continue
#        elif p_cm == 1:
#          if p > 5:
#            continue
#        elif p_cm == 2:
#          if p > 6:
#            continue
#        elif p_cm == 3:
#          if p > 7:
#            continue
#        elif p_cm == 4:
#          if p > 4:
#            continue

        cgs = group.TOhCG(p_cm, i, j, groups, ir1=_i1, ir2=_i2)

        # TODO: irreps explizit angeben. TOh gibt Liste der beitragenden irreps 
        # zurueck TOh.subduction_SU2(j) mit j = 2j+1
        #cgs = group.TOhCG(0, p, p, groups, ir1="A2g", ir2="T2g")
        #print("pandas")
        df = pd.concat([df, cgs.to_pandas()], ignore_index=True)
      except RuntimeError:
        continue

  df.rename(columns={'row' : '\mu', 'multi' : 'mult', 
                                       'cg' : 'cg-coefficient'}, inplace=True)
  df['cg-coefficient'] = df['cg-coefficient'].apply(aeval)


  # Create new column 'p' with tuple of momenta 
  # ( (p1x, p1y, p1z), (p2x, p2y, p2z) )
  # TODO warning for imaginary parts
  def to_tuple(list, sign=+1):
      return tuple([int(sign*l.real) for l in list])
  df['p1'] = df['p1'].apply(to_tuple)
  df['p2'] = df['p2'].apply(to_tuple)
  df['p'] = list(zip(df['p1'], df['p2']))
  df.drop(['p1', 'p2', 'ptot'], axis=1, inplace=True)

  # Inserting J, M to merge with basis and obtain gamma structure
  # Hardcoded: Scattering of two (pseudo)scalars (|J,M> = |0,0>)
  df['J'] = [(0,0)]*len(df.index) 
  df['M'] = [(0,0)]*len(df.index)

#  print df[((df['p'] == tuple([(0,-1,0),(0,1,1)])) | (df['p'] == tuple([(0,1,1),(0,-1,0)])) | (df['p'] == tuple([(0,1,0),(0,-1,1)])) | (df['p'] == tuple([(0,-1,1),(0,1,0)])) | (df['p'] == tuple([(-1,0,0),(1,0,1)])) | (df['p'] == tuple([(1,0,1),(-1,0,0)])) | (df['p'] == tuple([(1,0,0),(-1,0,1)])) | (df['p'] == tuple([(-1,0,1),(1,0,0)]))) & df['cg-coefficient'] != 0]
#  print df['Irrep'].unique()
#  print df[(df['Irrep'] == 'Ep1g')]

  df = select_irrep(df, irrep)

  return df

def get_lattice_basis(p_cm, verbose=True):
  """
  Calculate basis for irreducible representations of appropriate little group 
  of rotational symmetry for lattice in a moving reference frame with 
  momentum p_cm

  Parameters
  ----------
    p_cm : int, {0,1,2,3,4}
        Center of mass momentum of the lattice. Used to specify the appropriate
        little group of rotational symmetry. Absolute value of an integer 
        3-vector

  Returns
  -------
    df : pd.DataFrame
        Contains subduction coefficients for going from continuum to discete
        space. 
        Has columns Irrep, mult, J, M, cg-coefficient, p, \mu and unnamed 
        indices
  """

  prefs = [[0.,0.,0.], [0.,0.,1.], [0.,1.,1.], [1.,1.,1.]]

  # initialize groups
  S = 1./np.sqrt(2.)

  # tells clebsch_gordan to use cartesian basis
  U3 = np.asarray([[0,0,-1.],[1.j,0,0],[0,1,0]])
  U2 = np.asarray([[S,S],[1.j*S,-1.j*S]])
#  U3 = np.identity(3)
#  U2 = np.identity(2)

  try:
      groups = group.TOh.read(p2=p_cm)
      if not np.allclose(groups.U3, U3) or not np.allclose(groups.U2, U2):
          raise IOError("redo computation")
  except IOError:
      groups = group.TOh(pref=prefs[p_cm], irreps=True, U3=U3, U2=U2)
#      groups.save()

  # calc coefficients
  basis = group.TOhBasis(groups,jmax=2)
  # Isospin 1 hardcoded here
  df = basis.to_pandas(1)

  # munging to have a consistent format
  df.rename(columns={'row' : '\mu', 'coeff' : 'cg-coefficient'}, inplace=True)
  df['cg-coefficient'] = df['cg-coefficient'].apply(aeval)
  def to_tuple(list, sign=+1):
      return tuple([sign*int(l).real for l in list])
  df['p'] = df['p'].apply(to_tuple)

  if verbose:
    print 'lattice_basis'
    print df

  return df

# TODO: properly read that from infile and pass to get_clebsch_gordan
# TODO: actually use names to restrict basis_table to what was in the infile
def get_continuum_basis(names, basis_type, verbose):
  """
  Get table with chosen operators to transform as the *continuum* eigenstates
  of the vector spin representation

  Parameters
  ----------
  names : list of string
      Contains the names of chosen multiplets as the set of continuum 
      eigenstates is ambiguous and multiple choices might be wanted for a gevp.
  basis_type : string {cartesian, cyclic, cartesian-xzy, cyclic-xzy}

  Returns
  -------
  basis_table : pd.DataFrame
      Table with linear coefficients to contruct a spin 1 eigenstate from 
      operators. Rows are hierarchical containing the `name` of a multiplet and 
      the eigenstates it contains. There one column for each linearlz 
      independent Lorentz structure.
  """

  # implement trivial cartesian basis as it is taken account for in 
  # cg coefficients
  # TODO: Swithc case for the different basis types chosen for p=0,1,...
  sqrt2 = np.sqrt(2)
  sqrt3 = np.sqrt(3)
  sqrt6 = np.sqrt(6)
  if basis_type == "cartesian":
    ladder_operators = [[1, 0, 0], 
                        [0, 1, 0], 
                        [0, 0, 1]]
  elif basis_type == "cyclic":
    ladder_operators = [[1./sqrt2, -1j/sqrt2, 0], 
                        [0,         0,        1], 
                        [1./sqrt2, +1j/sqrt2, 0]]
  elif basis_type == "cyclic-christian":
    ladder_operators = [[1./sqrt2, -1j/sqrt2, 0], 
                        [0,         0,        -1.], 
                        [-1./sqrt2, -1j/sqrt2, 0]]
  elif basis_type == "cyclic-i":
    ladder_operators = [[1./sqrt2,  -1j/sqrt2, 0], 
                        [0,         0,        -1j], 
                        [-1./sqrt2, -1j/sqrt2, 0]]
  elif basis_type == "cartesian-xzy":
    ladder_operators = [[1, 0, 0], 
                        [0, 0, 1], 
                        [0, -1, 0]]
  elif basis_type == "cartesian-yzx":
    ladder_operators = [[0, 1, 0], 
                        [0, 0, 1], 
                        [1, 0, 0]]
  elif basis_type == "cyclic-xzy":
    ladder_operators = [[1./sqrt2,  0, 1./sqrt2], 
                        [0,         1, 0], 
                        [1./sqrt2,  0, -1./sqrt2]]
#  elif basis_type == "p2":
#    ladder_operators = [[(1./2+sqrt2),  1., 1./2], 
#                        [-1j/sqrt2,     0., -1j/sqrt2], 
#                        [(1./2-sqrt2),  1., 1./2]]
  elif basis_type == "p2":
    ladder_operators = [[ sqrt2,  1./2,     1./2], 
                        [    0.,  1j/sqrt2, -1j/sqrt2], 
                        [-sqrt2,  1./2,     1./2]]
  elif basis_type == "p3":
    ladder_operators = [[+1./2+sqrt3/6, 1./2+sqrt3/6, 1./sqrt3],
                        [-1./sqrt6*(1+1j), 1./sqrt6*(1+1j), -1./sqrt6*(1+1j)], 
                        [-(-1./2-sqrt3/6)*1j, (1./2-sqrt3/6)*1j, -1j/sqrt3]]
  elif basis_type == "ssh":
    ladder_operators = [[0, -1./sqrt2, 0], 
                        [0, 0, 1j], 
                        [-1j/sqrt2, 0, 0]]
  else:
    print "In get_continuum_basis: continuum_basis type ", basis_type, " not known!"
    return

  basis = np.array(ladder_operators).flatten()
#  basis = np.array([m + [0]*3 for m in ladder_operators] + \
#                                  [[0]*3+m for m in ladder_operators]).flatten()
  # hardcode basis operators for \gamma_i and \gamma_5\gamma_0\gamma_i
  # TODO: replace first list in MultiIndex.from_product by names (or some kind 
  # of latex(names)
#  basis_table = DataFrame(basis, \
#            index=pd.MultiIndex.from_product( \
#                [["\gamma_{i}  ", "\gamma_{50i}"], [1], [-1,0,1], \
#                                     [(1,), (2,), (3,), (13,), (14,), (15,)]], \
#                names=["gevp", 'J', 'M', '\gamma']), \
#            columns=['subduction-coefficient'], dtype=complex).sort_index()
  basis_table = DataFrame(basis, \
            index=pd.MultiIndex.from_product( \
                [["\gamma_{i}  "], [1], [-1,0,1], \
                                     [(1,), (2,), (3,)]], \
                names=["gevp", 'J', 'M', '\gamma']), \
            columns=['subduction-coefficient'], dtype=complex).sort_index()
  # conatenate basis chosen for two-pion operator. Trivial, because just two
  # singlet states.
  basis_table = pd.concat([basis_table, DataFrame([1], \
            index=pd.MultiIndex.from_product( \
                [["(\gamma_{5}, \gamma_{5})"], [(0,0)], [(0,0)], [(5,5)]], \
                names=["gevp", 'J', 'M', '\gamma']), \
            columns=['subduction-coefficient'], dtype=complex).sort_index()])

  if verbose:
    print 'basis_table'
    print basis_table

  return basis_table[basis_table['subduction-coefficient'] != 0]


# TODO: change p_cm to string symmetry_group in get_coeffients
# TODO: find a better name for diagram after Wick contraction
# TODO:  The information in irrep, mult and basis is redundant. return_cg() 
#        should be changed to simplify the interface
def get_coefficients(diagram, gammas, p_cm, irrep, basis, continuum_basis, \
                                                                       verbose):
  """
  Read table with required coefficients from forming continuum basis states, 
  subduction to the lattice and Clebsch-Gordan coupling

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
  basis : pd.DataFrame      
      discrete basis states restricted to *irrep* and *mult*. 
      Has columns J, M, cg-coefficient, p, \mu and unnamed indices
  continuum_basis : string
      String specifying the continuum basis to be chosen. 

  Returns
  ------- 
  coefficients_irrep: pd.DataFrame
      Table with all quantum numbers on source and sink as columns and the rows
      of the irreducible representations as rows. Also contains two columns
      with the appropriate Clebsch Gordan coefficient for all these quantum
      numbers
  """

  basis = select_irrep(basis, irrep)
#  if irrep == 'A1g':
#    irrep = 'A2g'
#    mult = 3

  if diagram.startswith('C2'):
    cg_table_so = basis
    cg_table_si = basis.copy()
    cg_table_si['p'] = ((-1)*cg_table_si['p'].apply(np.array)).apply(tuple)
  elif diagram.startswith('C3'):
    # get factors for the desired irreps
    cg_one_operator = basis
    cg_two_operators = return_cg(p_cm, irrep)
    # for 3pt function we have pipi operator at source and rho operator at sink
    cg_table_so, cg_table_si = cg_two_operators, cg_one_operator
    cg_table_si['p'] = (cg_table_si['p'].apply(np.array)*(-1)).apply(tuple)
  elif diagram.startswith('C4'):
    cg_table_so = return_cg(p_cm, irrep)
    cg_table_si = cg_table_so.copy()
    def to_tuple(list):
      return tuple([tuple(l) for l in list])
    cg_table_si['p'] = (cg_table_si['p'].apply(np.array)*(-1)).apply(to_tuple)

  else:
    print 'in get_coefficients: diagram unknown! Quantum numbers corrupted.'
    return

  # express basis states for all Lorentz structures in `gammas` in terms of 
  # physical Dirac operators
  continuum_basis_table = get_continuum_basis(gammas, continuum_basis, verbose)

  # express the subduced eigenstates in terms of Dirac operators.
  cg_table_so = pd.merge(cg_table_so, continuum_basis_table.reset_index()).\
                                                   set_index('\mu').sort_index()   
  cg_table_si = pd.merge(cg_table_si, continuum_basis_table.reset_index()).\
                                                   set_index('\mu').sort_index()  
  # Munging the result: Delete rows with coefficient 0, combine coefficients 
  # and clean columns no longer needed.
  cg_table_so = cg_table_so[cg_table_so['cg-coefficient'] != 0]
  cg_table_so['coefficient'] = \
           cg_table_so['cg-coefficient'] * cg_table_so['subduction-coefficient']
  cg_table_so.drop(['J', 'M', 'cg-coefficient', 'subduction-coefficient'], \
                                                           axis=1, inplace=True)
  cg_table_si = cg_table_si[cg_table_si['cg-coefficient'] != 0]
  cg_table_si['coefficient'] = \
           cg_table_si['cg-coefficient'] * cg_table_si['subduction-coefficient']

  cg_table_si.drop(['J', 'M', 'cg-coefficient', 'subduction-coefficient'], \
                                                           axis=1, inplace=True)

  # combine clebsch-gordan coefficients for source and sink into one DataFrame
  coefficients_irrep = pd.merge(cg_table_so, cg_table_si, how='inner', \
      left_index=True, right_index=True, suffixes=['_{so}', '_{si}']) 

  if verbose:
    print 'coefficients_irrep'
    print coefficients_irrep

  return coefficients_irrep

def set_lookup_qn_irrep(coefficients_irrep, qn, verbose):
  """
  Calculate table with all required coefficients and quantum numbers of the 
  states needed to obtain eigenstates under a certain irreducible representation

  Parameters
  ----------  
  coefficients_irrep: pd.DataFrame
      Table with all quantum numbers on source and sink as columns and the rows
      of the irreducible representations as rows. Also contains two columns
      with the appropriate Clebsch Gordan coefficient for all these quantum
      numbers

  qn : pd.DataFrame
      pd.DataFrame with every row being a set of physical quantum numbers
      contributing to `diagram`. The rows are indexed by a unique identifier

  Returns
  ------- 
  qn_irrep : pd.DataFrame
      Table with a column for each quantum number at source and sink, the
      Clebsch-Gordan coefficients at source and sink and a column decoding
      the row and column of the gevp these quantum numbers enter into:
      \mu \gamma_{so} cg-coefficient_{so} p_{so} \gamma_{si} \
          cg-coefficient_{si} p_{si} index gevp_row gevp_col
  """

  # associate clebsch-gordan coefficients with the correct qn index

#  print coefficients_irrep[coefficients_irrep['p_{so}'] == tuple([(2,0,0),(-1,0,0)])] 
#  print qn[qn['p_{so}'] == tuple([(0,0,0),(0,0,1)])]
#  return
  qn_irrep = pd.merge(coefficients_irrep.reset_index(), qn.reset_index())
#  print 'qn_irrep'
#  print qn_irrep[qn_irrep['p_{so}_x'] == qn_irrep['p_{so}_y']]

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
#  del(qn_irrep['mult_{so}'])
#  del(qn_irrep['mult_{si}'])

  if verbose:
    print 'qn_irrep'
    print qn_irrep
    utils.write_hdf5_correlators('./', 'qn_irrep.h5', qn_irrep, 'data', verbose=False)

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
                      'p_{so}', '\gamma_{so}', 'p_{si}', '\gamma_{si}', 'mult_{so}', 'mult_{si}'])
  subduced = subduced.ix[:,2:].multiply(subduced['coefficient_{so}']*
                               np.conj(subduced['coefficient_{si}']), axis=0)

  subduced.columns=pd.MultiIndex.from_tuples(subduced.columns, \
                                                         names=('cnfg', 'T'))

  subduced = subduced.sort_index()

  # I do not know why the dtype got converted to object, but convert it back
  # to complex
  return subduced.apply(pd.to_numeric).sort_index()

