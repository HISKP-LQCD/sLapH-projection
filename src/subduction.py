#!/usr/bin/python

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import itertools as it
import cmath
import functools
import os

from utils import _scalar_mul, _abs2, _minus

import operator
import collections

from asteval import Interpreter
aeval = Interpreter()
aeval.symtable['I'] = 1j

from ast import literal_eval

#import clebsch_gordan_2pt as cg_2pt
#import clebsch_gordan_4pt as cg_4pt
import utils

#from clebsch_gordan import group

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
        df restricted to *irrep* and *mult*=1. 
        Has columns p, J, M, coefficient, index \mu

  See
  ---
    read_sc()
  """

  return select_irrep_mult(df, irrep, 1)


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
        Has columns p, J, M, coefficient, index \mu

  See
  ---
    read_sc()
  """

  return df.xs((irrep,mult), level=('Irrep','mult'), drop_level=False)

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

  # we want all possible irreps for pipi, only the combinations possible for
  # C2 for the rho
  #df = select_irrep(df, irrep)

  return df

def read_sc_2(p_cm_vecs, path, verbose=True, j=1):
  """
  Read subduction coefficients from SO(3) to irreducible representations of 
  appropriate little group of rotational symmetry for lattice in a reference 
  frame moving with momentum p_cm \in list_p_cm

  Parameters
  ----------
    p_cm_vecs : list
        Center of mass momentum of the lattice. Used to specify the appropriate
        little group of rotational symmetry. Contains integer 3-vectors
    path : string
        Path to files with subduction coefficients

  Returns
  -------
    df : pd.DataFrame
        Contains subduction coefficients for going from continuum to discete
        space. 
        Has columns Irrep, mult, J, M, coefficient, p, \mu and unnamed 
        indices

  Note
  ----
    Filename of subduction coefficients hardcoded. Expected to be 
    "J%d-P%1i%1i%1i-operators.txt"
    # "lattice-basis_J%d_P%1i%1i%1i_Msum.dataframe"
  """

  subduction_coefficients = DataFrame()

  for p_cm_vec in p_cm_vecs:
    name = path +'/' + 'J{0}-P{1}-operators.txt'.format(\
           j, "".join([str(p) for p in eval(p_cm_vec)]))

    if not os.path.exists(name):
      print 'Warning: Could not find {}'.format(name)
      continue

    df = pd.read_csv(name, sep="\t", dtype=str)
    df.rename(columns=lambda x: x.strip(), inplace=True)

    df['p_{cm}'] = [p_cm_vec] * len(df)
    df.rename(columns={'alpha' : '\mu'}, inplace=True)
    del df['beta']
    df['mult'] = 1
    df = df.set_index(['p_{cm}', 'Irrep', '\mu', 'mult'])

    df['coefficient'] = df['coefficient'].apply(aeval)
    df['p^1'] = df['p^1'].apply(literal_eval).apply(str)
    df['p^2'] = df['p^2'].apply(literal_eval).apply(str)
    df['q'] = df['q'].apply(literal_eval).apply(str)
    df.rename(columns={'p^1' : 'p^{0}', 'p^2' : 'p^{1}'}, inplace=True)
    df = df[(df['p^{0}'] != str((0,0,0))) | (df['p^{1}'] != str((0,0,0)))]
    del df['abs(p1)']
    del df['abs(p2)']

    df['J^{0}'] = 0
    df['J^{1}'] = 0
    df['M^{0}'] = 0
    df['M^{1}'] = 0

    if verbose:
      print 'subduction_coefficients for {}'.format(p_cm_vec)
      print df, '\n'

    subduction_coefficients = pd.concat([subduction_coefficients, df])

  return subduction_coefficients.sort_index()


# TODO: Write a function to calculate cross product if basis is not 
#       complete and orthonormalize basis states
# To extend to 2 particle operators this must support use of 2 momenta as well.
# We get an additional column \vec{q} that works similar to the row of the 
# irrep. Either the maple script has to find unique operators, or an additional
# unique function must be used here. I prefer the former.
def read_sc(p_cm_vecs, path, verbose=True, j=1):
  """
  Read subduction coefficients from SO(3) to irreducible representations of 
  appropriate little group of rotational symmetry for lattice in a reference 
  frame moving with momentum p_cm \in list_p_cm

  Parameters
  ----------
    p_cm_vecs : list
        Center of mass momentum of the lattice. Used to specify the appropriate
        little group of rotational symmetry. Contains integer 3-vectors
    path : string
        Path to files with subduction coefficients

  Returns
  -------
    df : pd.DataFrame
        Contains subduction coefficients for going from continuum to discete
        space. 
        Has columns Irrep, mult, J, M, coefficient, p, \mu and unnamed 
        indices

  Note
  ----
    Filename of subduction coefficients hardcoded. Expected to be 
    "J%d-P%1i%1i%1i-operators.txt"
    # "lattice-basis_J%d_P%1i%1i%1i_Msum.dataframe"
  """

  subduction_coefficients = DataFrame()

  for p_cm_vec in p_cm_vecs:

#    name = path +'/' + 'lattice-basis_J{0}_P{1}_Msum.dataframe'.format(\
#           j, "".join([str(p) for p in eval(p_cm_vec)]))
    name = path +'/' + 'J{0}-P{1}-operators.txt'.format(\
           j, "".join([str(p) for p in eval(p_cm_vec)]))

    if not os.path.exists(name):
      print 'Warning: Could not find {}'.format(name)
      continue
    df = pd.read_csv(name, delim_whitespace=True, dtype=str) 

    df = pd.merge(df.ix[:,2:].stack().reset_index(level=1), df.ix[:,:2], 
                  left_index=True, 
                  right_index=True)

    # Munging of column names
    df.columns = ['M^{0}', 'coefficient', 'Irrep', '\mu']
    df['mult'] = 1
    df['p_{cm}'] = [p_cm_vec] * len(df)
    df['coefficient'] = df['coefficient'].apply(aeval)
    df['M^{0}'] = df['M^{0}'].apply(int)
    df = df.set_index(['p_{cm}', 'Irrep', '\mu', 'mult'])
    df['p^{0}'] = [p_cm_vec] * len(df)
    df['J^{0}'] = j
    df = df[['p^{0}','J^{0}','M^{0}','coefficient']]

    if verbose:
      print 'subduction_coefficients for {}'.format(p_cm_vec)
      print df, '\n'

    subduction_coefficients = pd.concat([subduction_coefficients, df])

  return subduction_coefficients.sort_index()

# TODO: properly read that from infile and pass to get_clebsch_gordan
# TODO: actually use names to restrict basis_table to what was in the infile
# TODO: Unify that with set_lookup_g
# Bug: flag for wanted gamma structures is not used
def set_continuum_basis(names, basis_type, verbose):
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
      the eigenstates it contains. There one column for each linearly
      independent Lorentz structure.
  """

  basis_J0 = DataFrame({'J' : [0],
                        'M' : [0], 
                        'gamma_id' : range(1)*1, 
                        'coordinate' : [1]})

  gamma_5 = DataFrame({'\gamma' : [5],
                         'operator_label' : '\gamma_{5}'})

  gamma = pd.concat([eval(n) for n in names[0]])

  basis_J0 = pd.merge(basis_J0, gamma, how='left', left_on=['gamma_id'], right_index=True)
  del(basis_J0['gamma_id'])
  basis_J0 = basis_J0.set_index(['J','M'])

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
    # Standard ladder operators; default choice
    ladder_operators = [[1./sqrt2, -1j/sqrt2, 0], 
                        [0,         0,        1], 
                        [1./sqrt2, +1j/sqrt2, 0]]
  elif basis_type == "cyclic-christian":
    ladder_operators = [[1./sqrt2, -1j/sqrt2, 0], 
                        [0,         0,        -1.], 
                        [-1./sqrt2, -1j/sqrt2, 0]]
  elif basis_type == "marcus-cov":
    ladder_operators = [[1/sqrt2,   -1j/sqrt2, 0], 
                        [0,          0,        1], 
                        [-1/sqrt2,  -1j/sqrt2,  0]]
  elif basis_type == "marcus-con":
    ladder_operators = [[1/sqrt2,   1j/sqrt2, 0], 
                        [0,          0,        1], 
                        [-1/sqrt2,  1j/sqrt2,  0]]
  elif basis_type == "dudek":
    ladder_operators = [[1j/sqrt2,  -1./sqrt2, 0], 
                        [0,         0,         1j], 
                        [-1j/sqrt2, -1./sqrt2, 0]]
  else:
    print "In get_continuum_basis: continuum_basis type ", basis_type, " not known!"
    exit()

  basis_J1 = DataFrame({'J' : [1]*9,
                        'M' : [-1]*3+[0]*3+[1]*3, 
                        'gamma_id' : range(3)*3, 
                        'coordinate' : np.array(ladder_operators).flatten()})

  gamma_i   = DataFrame({'\gamma' : [1,2,3],
                         'operator_label' : '\gamma_{i}'})
  gamma_50i = DataFrame({'\gamma' : [13,14,15],
                         'operator_label' : '\gamma_{50i}'})

  gamma = pd.concat([eval(n) for n in names[1]])

  basis_J1 = pd.merge(basis_J1, gamma, how='left', left_on=['gamma_id'], right_index=True)
  del(basis_J1['gamma_id'])
  basis_J1 = basis_J1.set_index(['J','M'])

  basis = pd.concat([basis_J0, basis_J1])

  if verbose:
    print 'basis'
    print basis

  return basis[basis['coordinate'] != 0]


# TODO: find a better name for diagram after Wick contraction
# TODO: No restriction to multiplicacy currently done
def project_operators(di, sc, sc_2, continuum_operators, verbose):
  """
  Project continuum operators to lattice using subduction coefficients 

  Parameters
  ----------  
  di : namedtuple('ContractionType', ['diagram', 'irrep'])
      Specifies the type of operator and lattice
      diagram : string, {'C2', 'C3', 'C4'}
          Type of correlation function contributing to gevp. The number is the 
          number of quarklines
      irrep : string
          Name of the irreducible representation of the little group the operator
          is required to transform under.
  sc : pd.DataFrame      
      Subduction coefficients. Has column coefficient and MultiIndex 
      p_{cm} Irrep \mu mult
  continuum_operators: pd.DataFrame
      Basis of SO(3) eigenstates (Spin-J basis). Has columns 
      coordinate \gamma gevp and MultiIndex J M 

  Returns
  ------- 
  lattice_operators: pd.DataFrame
      Table with all quantum numbers on source and sink as columns and the rows
      of the irreducible representations as rows. Also contains two columns
      with the appropriate Clebsch Gordan coefficient for all these quantum
      numbers
  """

  # Restrict subduction coefficients to irredicible representation specified 
  # in di


  if di.diagram.startswith('C2'):
    continuum_labels_so = [['J^{0}','M^{0}']]
    operator_so = select_irrep(sc, di.irrep)

    continuum_labels_si = [['J^{0}','M^{0}']]
    operator_si = select_irrep(sc, di.irrep)
#    operator_si['p'] = ((-1)*operator_si['p'].apply(np.array)).apply(tuple)
  elif di.diagram.startswith('C3'):
    # for 3pt function we have pipi operator at source and rho operator at sink
    continuum_labels_so = [['J^{0}','M^{0}'], ['J^{1}','M^{1}']]
    operator_so = select_irrep(sc_2, di.irrep)

    continuum_labels_si = [['J^{0}','M^{0}']]
    operator_si = select_irrep(sc, di.irrep)
#    operator_si['p'] = (operator_si['p'].apply(np.array)*(-1)).apply(tuple)
  elif di.diagram.startswith('C4'):
    operator_so = return_cg(p_cm, irrep)
    operator_si = operator_so.copy()
#    def to_tuple(list):
#      return tuple([tuple(l) for l in list])
#    operator_si['p'] = (operator_si['p'].apply(np.array)*(-1)).apply(to_tuple)

  else:
    print 'in get_coefficients: diagram unknown! Quantum numbers corrupted.'
    return

  # Project operators 
  # :math: `O^{\Gamma, \mu} = S^{\Gamma J}_{\mu, m} \cdot O^{J, m}`
  for cl_so in continuum_labels_so:
    operator_so = pd.merge(operator_so, continuum_operators, 
                           how='left', left_on=cl_so, right_index=True,
                           suffixes=['^{0}', '^{1}']) 
    # Combine coefficients and clean columns no longer needed: Labels for continuum 
    # basis and factors entering the final coefficient
    operator_so['coefficient'] = \
             operator_so['coefficient'] * operator_so['coordinate']
    operator_so.drop(cl_so + ['coordinate'],
                     axis=1, inplace=True)

  for cl_si in continuum_labels_si:
    operator_si = pd.merge(operator_si, continuum_operators, 
                           how='left', left_on=cl_si, right_index=True,
                           suffixes=['^{0}', '^{1}']) 
    # Combine coefficients and clean columns no longer needed: Labels for continuum 
    # basis and factors entering the final coefficient
    operator_si['coefficient'] = \
             operator_si['coefficient'] * operator_si['coordinate']
    operator_si.drop(cl_si + ['coordinate'],
                     axis=1, inplace=True)

  # Rename if merge has not appended a suffix
  operator_so.rename(columns={'\gamma' : '\gamma^{0}', 'p^{1}' : 'p^{1}_{so}'}, inplace=True)
  operator_so['operator_label'] = operator_so[[col for col in operator_so.columns if 'label' in col]].apply(lambda x: ', '.join(x), axis=1)
  operator_so.drop(['operator_label^{0}', 'operator_label^{1}'], 
                    axis=1, inplace=True, errors='ignore')
  operator_si.rename(columns={'\gamma' : '\gamma^{0}', 'p^{1}' : 'p^{1}_{so}'}, inplace=True)
  operator_si['operator_label'] = operator_si[[col for col in operator_si.columns if 'label' in col]].apply(lambda x: ', '.join(x), axis=1)
  operator_si.drop(['operator_label^{0}', 'operator_label^{1}'], 
                    axis=1, inplace=True, errors='ignore')

  # combine clebsch-gordan coefficients for source and sink into one DataFrame
  operator_so.rename(columns={'\gamma^{0}' : '\gamma^{0}_{so}', 
                              '\gamma^{1}' : '\gamma^{1}_{so}',
                              'q' : 'q_{so}'}, inplace=True)
  operator_si.rename(columns={'\gamma^{0}' : '\gamma^{0}_{si}', 
                              '\gamma^{1}' : '\gamma^{1}_{si}'}, inplace=True)
  lattice_operators = pd.merge(operator_so, operator_si, 
                               how='inner', left_index=True, right_index=True, 
                               suffixes=['_{so}', '_{si}']) 

  lattice_operators['coefficient'] = lattice_operators['coefficient_{so}'].apply(np.conj) \
                                        * lattice_operators['coefficient_{si}']
  lattice_operators.drop(['coefficient_{so}', 'coefficient_{si}'], axis=1, inplace=True)

  lattice_operators.reset_index(inplace=True)
  index = lattice_operators.columns.difference(['coefficient']).tolist()
  order = { 'Irrep' : 0, 
            'mult' : 1,             
            'p_{cm}' : 2, 
            '\mu' : 3, 
            'p^{0}_{so}' : 4, 
            'p^{1}_{so}' : 5, 
            'p^{0}_{si}' : 6, 
            'p^{1}_{si}' : 7, 
            'q_{so}' : 8, 
            'q_{si}' : 9, 
            '\gamma^{0}_{so}' : 10, 
            '\gamma^{1}_{so}' : 11, 
            '\gamma^{0}_{si}' : 12, 
            '\gamma^{1}_{si}' : 13,
            'operator_label_{so}' : 14,
            'operator_label_{si}' : 15} 
  index = sorted(index, key=lambda x : order[x])
  lattice_operators.set_index(index, inplace=True)

  print lattice_operators.sort_index()

  lattice_operators = lattice_operators.sum(axis=0, level=index)

  print lattice_operators.sort_index()

  # Munging the result: Delete rows with coefficient 0, 
  lattice_operators = lattice_operators[lattice_operators['coefficient'] != 0]

  if verbose:
    print 'lattice_operators'
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print lattice_operators

  return lattice_operators

def set_lookup_corr(coefficients_irrep, qn, verbose):
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
  lookup_corr: pd.DataFrame
      Table with a column for each quantum number at source and sink, the
      Clebsch-Gordan coefficients at source and sink and a column decoding
      the row and column of the gevp these quantum numbers enter into:
      \mu \gamma_{so} coefficient_{so} p_{so} \gamma_{si} \
          coefficient_{si} p_{si} index gevp_row gevp_col
  """

  # associate clebsch-gordan coefficients with the correct qn index
  # TODO: Check whether left merge results in nan somewhere as in this case data is missing
  lookup_corr = pd.merge(coefficients_irrep.reset_index(), qn.reset_index(), 
                      how='left')


  # Add two additional columns with the same string if the quantum numbers 
  # describe equivalent physical constellations: gevp_row and gevp_col
  lookup_corr['gevp_row'] = 'p: ' + lookup_corr['p_{cm}'].apply(eval).apply(utils._abs2).apply(str) + \
                              ', g: ' + lookup_corr['operator_label_{so}']
  lookup_corr['gevp_col'] = 'p: ' + lookup_corr['p_{cm}'].apply(eval).apply(utils._abs2).apply(str) + \
                              ', g: ' + lookup_corr['operator_label_{si}']

  lookup_corr.drop(['operator_label_{so}', 'operator_label_{si}'], axis=1, inplace=True)

  # Set index as it shall appear in projected correlators
  index = lookup_corr.columns.difference(['index', 'coefficient']).tolist()
  order = { 'Irrep' : 0, 
            'mult' : 1,             
            'gevp_row' : 2, 
            'gevp_col' : 3, 
            'p_{cm}' : 4, 
            '\mu' : 5, 
            'p^{0}_{so}' : 6, 
            'p^{1}_{so}' : 7, 
            'p^{0}_{si}' : 8, 
            'p^{1}_{si}' : 9, 
            'q_{so}' : 10, 
            'q_{si}' : 11, 
            '\gamma^{0}_{so}' : 12, 
            '\gamma^{1}_{so}' : 13, 
            '\gamma^{0}_{si}' : 14, 
            '\gamma^{1}_{si}' : 15} 
  index = sorted(index, key=lambda x : order[x])
  lookup_corr.set_index(index, inplace=True)

  if verbose:
    print 'lookup_corr'
    print lookup_corr
    utils.write_hdf5_correlators('./', 'lookup_corr.h5', lookup_corr, 'data', verbose=False)

  return lookup_corr


def project_correlators(data, qn_irrep):
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
      \mu \gamma_{so} coefficient_{so} p_{so} \gamma_{si} \
          coefficient_{si} p_{si} index gevp_row gevp_col

  Returns
  -------
  subduced : pd.DataFrame
      Table with physical quantum numbers as rows and gauge configuration and 
      lattice time as columns.
      Contains the linear combinations of correlation functions transforming
      like what the parameters qn_irrep was created with
  """

  # actual subduction step. sum conj(cg_so) * cg_si * corr
  projected_correlators = pd.merge(qn_irrep, data.T, 
                                   how='left', left_on=['index'], right_index=True)
  del projected_correlators['index']

  projected_correlators = projected_correlators[
        projected_correlators.columns.difference(['coefficient'])].\
       multiply(projected_correlators['coefficient'], axis=0)

  projected_correlators.columns=pd.MultiIndex.from_tuples(projected_correlators.columns, \
                                                         names=('cnfg', 'T'))

  projected_correlators = projected_correlators.sort_index()

  # I do not know why the dtype got converted to object, but convert it back
  # to complex
  return projected_correlators.apply(pd.to_numeric).sort_index()

