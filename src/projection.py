import numpy as np
import pandas as pd
from pandas import Series, DataFrame

from projection_lattice_basis import read_sc, read_sc_2

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


# The coefficients are read again for each diagram, but that is not the 
# performance-critical part
def read_lattice_basis(p_cm_vecs, path_one_particle_coeffs, path_two_particle_coeffs, 
        j=1, verbose=True):

  # Read subduction coefficients for all momenta in list_p_cm
  sbdctn_coeffs = read_sc(p_cm_vecs, path_one_particle_coeffs, verbose, j=j)

  sbdctn_coeffs_2 = read_sc_2(p_cm_vecs, path_two_particle_coeffs, verbose, j=j)

  return sbdctn_coeffs, sbdctn_coeffs_2

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


