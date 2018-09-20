import numpy as np
import pandas as pd
from pandas import Series, DataFrame

# Used in isospin
from ast import literal_eval

from projection_interface_maple import read_sc, read_sc_2
import utils


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

    return df.xs((irrep, mult), level=('Irrep', 'mult'), drop_level=False)


def get_list_of_irreps(p_cm_vecs, path_to_one_particle_coeffs, j):
    """
    Get list of all irreducible representations appearing in the subduction step

    Parameters
    ----------
      p_cm_vecs : list
          Center of mass momentum of the lattice. Used to specify the appropriate little
          group of rotational symmetry. Contains integer 3-vectors
      path_to_one_particle_coeffs : string
          Path to files with subduction coefficients
      j: int
          Angular momentum quantum number defining representation of SO(3) from which to
          subduce

    Returns
    -------
      List with the names of all contributing lattice irreducible representations
    """

    sbdctn_coeffs = read_sc(p_cm_vecs, path_to_one_particle_coeffs, verbose=0, j=j)

    return sbdctn_coeffs.index.get_level_values('Irrep').unique()

# The coefficients are read again for each diagram, but that is not the
# performance-critical part


def read_lattice_basis(correlator, p_cm_vecs, path_to_one_particle_coeffs,
                       path_to_two_particle_coeffs, j=1, verbose=1):
    """
    Distinguish by diagram whether one- or two-particle diagrams are needed at source and
    sink and read projection coefficients

    Parameters
    ----------
      diagram: string
          The name of the diagram from wick contractions.
      p_cm_vecs : list
          Center of mass momentum of the lattice. Used to specify the appropriate little
          group of rotational symmetry. Contains integer 3-vectors
      path_to_one_particle_coeffs : string
          Path to files with projection coefficients for one-particle operators
      path_to_two_particle_coeffs : string
          Path to files with projection coefficients for two-particle operators

    Returns
    -------
      pd.DataFrame, pd.DataFrame
          Tables with hierarchical MultiIndex ('p_{cm}', 'Irrep', '\mu', 'mult'),
          a column 'coefficient' and columns 'p', 'J' and 'M' for every particle
          and a column 'q' in case of two particles.

    Note
    ----
      The coefficients express the basis states of lattice irreducible representations as
      linear combination of angular momentum eigenstates.
    """

    # Read subduction coefficients for all momenta in list_p_cm
    if correlator == 'C2' or correlator == 'C3':
        sbdctn_coeffs = read_sc(p_cm_vecs, path_to_one_particle_coeffs, verbose, j=j)
    if correlator == 'C3' or correlator == 'C4':
        sbdctn_coeffs_2 = read_sc_2(p_cm_vecs, path_to_two_particle_coeffs, verbose, j=j)

    if correlator == 'C2':
        return sbdctn_coeffs, sbdctn_coeffs
    if correlator == 'C3':
        return sbdctn_coeffs_2, sbdctn_coeffs
    if correlator == 'C4':
        return sbdctn_coeffs_2, sbdctn_coeffs_2
    else:
        print 'in read_lattice_basis: correlator unknown! Quantum numbers corrupted.'
        return

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

    basis_J0 = DataFrame({'J': [0],
                          'M': [0],
                          'gamma_id': range(1) * 1,
                          'coordinate': [1]})

    gamma_5 = DataFrame({'\gamma': [5],
                         'operator_label': '\gamma_{5}'})
    gamma_0 = DataFrame({'\gamma': [0],
                         'operator_label': '\gamma_{0}'})

    gamma = pd.concat([eval(n) for n in names[0]])

    basis_J0 = pd.merge(
        basis_J0,
        gamma,
        how='left',
        left_on=['gamma_id'],
        right_index=True)
    del(basis_J0['gamma_id'])
    basis_J0 = basis_J0.set_index(['J', 'M'])

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
        ladder_operators = [[1. / sqrt2, -1j / sqrt2, 0],
                            [0, 0, 1],
                            [1. / sqrt2, +1j / sqrt2, 0]]
    elif basis_type == "cyclic-christian":
        ladder_operators = [[1. / sqrt2, -1j / sqrt2, 0],
                            [0, 0, -1.],
                            [-1. / sqrt2, -1j / sqrt2, 0]]
    elif basis_type == "marcus-cov":
        ladder_operators = [[1 / sqrt2, -1j / sqrt2, 0],
                            [0, 0, 1],
                            [-1 / sqrt2, -1j / sqrt2, 0]]
    elif basis_type == "marcus-con":
        ladder_operators = [[1 / sqrt2, 1j / sqrt2, 0],
                            [0, 0, 1],
                            [-1 / sqrt2, 1j / sqrt2, 0]]
    elif basis_type == "dudek":
        ladder_operators = [[1j / sqrt2, -1. / sqrt2, 0],
                            [0, 0, 1j],
                            [-1j / sqrt2, -1. / sqrt2, 0]]
    elif basis_type == "test":
        ladder_operators = [[-1j / sqrt2, -1 / sqrt2, 0],
                            [0, 0, 1],
                            [1j / sqrt2, -1 / sqrt2, 0]]
    else:
        print "In set_continuum_basis: continuum_basis type ", basis_type, " not known!"
        exit()

    basis_J1 = DataFrame({'J': [1] * 9,
                          'M': [-1] * 3 + [0] * 3 + [1] * 3,
                          'gamma_id': range(3) * 3,
                          'coordinate': np.array(ladder_operators).flatten()})

    gamma_i = DataFrame({'\gamma': [1, 2, 3],
                         'operator_label': '\gamma_{i}'})
    gamma_50i = DataFrame({'\gamma': [13, 14, 15],
                           'operator_label': '\gamma_{50i}'})

    gamma = pd.concat([eval(n) for n in names[1]])

    basis_J1 = pd.merge(
        basis_J1,
        gamma,
        how='left',
        left_on=['gamma_id'],
        right_index=True)
    del(basis_J1['gamma_id'])
    basis_J1 = basis_J1.set_index(['J', 'M'])

    basis = pd.concat([basis_J0, basis_J1])

    if verbose >= 1:
        print 'basis'
    if verbose == 1:
        print basis.head()
    if verbose >= 2:
        print basis

    return basis[basis['coordinate'] != 0]


# TODO: find a better name for diagram after Wick contraction
# TODO: No restriction to multiplicacy currently done
def project_continuum_to_lattice(correlator, operator_so, operator_si,
                      continuum_operators, verbose):
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

    if correlator == 'C2':
        continuum_labels_so = [['J^{0}', 'M^{0}']]
        continuum_labels_si = [['J^{0}', 'M^{0}']]
    elif correlator == 'C3':
        # for 3pt function we have pipi operator at source and rho operator at sink
        continuum_labels_so = [['J^{0}', 'M^{0}'], ['J^{1}', 'M^{1}']]
        continuum_labels_si = [['J^{0}', 'M^{0}']]
    elif correlator == 'C4':
        continuum_labels_so = [['J^{0}', 'M^{0}'], ['J^{1}', 'M^{1}']]
        continuum_labels_si = [['J^{0}', 'M^{0}'], ['J^{1}', 'M^{1}']]
    else:
        print 'in project_operators: correlator unknown! Quantum numbers corrupted.'
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

    # Munging for C3: Rename if merge has not appended a suffix
    operator_so.rename(
        columns={
            '\gamma': '\gamma^{0}',
            'p^{1}': 'p^{1}_{so}'},
        inplace=True)
    operator_si.rename(
        columns={
            '\gamma': '\gamma^{0}',
            'p^{1}': 'p^{1}_{si}'},
        inplace=True)

    operator_so.rename(columns={'\gamma^{0}': '\gamma^{0}_{so}',
                                '\gamma^{1}': '\gamma^{1}_{so}',
                                'q': 'q_{so}'}, inplace=True)
    operator_si.rename(columns={'\gamma^{0}': '\gamma^{0}_{si}',
                                '\gamma^{1}': '\gamma^{1}_{si}',
                                'q': 'q_{si}'}, inplace=True)

    # :warning: hardcoded factor
    # Correct sign error from sLapH-contractions
    operator_so.loc[operator_so[r'\gamma^{0}_{so}'].isin([0,1,2,3]), 'coefficient'] *= (-1)
    if r'\gamma^{1}_{so}' in operator_so.columns:
        operator_so.loc[operator_so[r'\gamma^{1}_{so}'].isin([0,1,2,3]), 'coefficient'] *= (-1)
    operator_si.loc[operator_si[r'\gamma^{0}_{si}'].isin([0,1,2,3]), 'coefficient'] *= (-1)
    if r'\gamma^{1}_{si}' in operator_si.columns:
        operator_si.loc[operator_si[r'\gamma^{1}_{si}'].isin([0,1,2,3]), 'coefficient'] *= (-1)

    return operator_so, operator_si

##########################################################################################

def project_angular(j, correlator, continuum_basis_string, gamma_input, list_of_pcm, 
            path_to_sc, path_to_sc_2, verbose):

    # Express :math:`O^{J,M}` in terms of physical Dirac operators
    # specified in `gamma_input`
    continuum_operators = set_continuum_basis(
        gamma_input, continuum_basis_string, verbose)

    # read coefficients for correlation function, little group
    # and irreducible representation given by correlator, p_cm
    # irrep and mult
    lattice_operators_so, lattice_operators_si = read_lattice_basis(
        correlator, list_of_pcm, path_to_sc, path_to_sc_2, j, verbose)

    operators_so, operators_si = project_continuum_to_lattice(
        correlator, lattice_operators_so, lattice_operators_si,
        continuum_operators, verbose)

    return operators_so, operators_si

# Create combinations of definite isospin
def project_isospin(operator_so, operator_si):

    label = 'q_{so}'
    if label in operator_so.columns:

        # sign from twisted mass rotation
#        operator_so.loc[operator_so[r'\gamma^{0}_{so}'].isin([0]), 'coefficient'] *= (-1)
        operator_so.loc[operator_so[r'\gamma^{1}_{so}'].isin([0]), 'coefficient'] *= (-1)

        operator_so[label] = operator_so[label].apply(literal_eval)
        
        isospin_neg = operator_so[operator_so[label] <= (0, 0, 0)]
        isospin_neg.loc[:, 'coefficient'] *= -1
#        isospin_neg.loc[(isospin_neg['\gamma^{0}_{so}'] == isospin_neg['\gamma^{1}_{so}']), 'coefficient'] *= -1
#        isospin_neg.rename(columns={'\gamma^{0}_{so}': '\gamma^{1}_{so}',
#                                    '\gamma^{1}_{so}': '\gamma^{0}_{so}'}, inplace=True)
#        isospin_neg.rename(columns={'operator_label^{0}': 'operator_label^{1}',
#                                    'operator_label^{1}': 'operator_label^{0}'}, inplace=True)
        isospin_neg.loc[:, label] = isospin_neg.loc[:, label].apply(utils._minus)

        isospin_pos = operator_so[operator_so[label] >= (0, 0, 0)]

        operator_so = pd.concat([isospin_pos, isospin_neg])
        operator_so[label] = operator_so[label].apply(str)
        operator_so['coefficient'] /= np.sqrt(2)

        # average g^0 g^5 and g^5 g^0 by flipping label for the former
        operator_so.loc[(operator_so['\gamma^{0}_{so}'] == 0) & (operator_so['\gamma^{1}_{so}'] == 5), ['operator_label^{0}', 'operator_label^{1}']] = operator_so.loc[(operator_so['\gamma^{0}_{so}'] == 0) & (operator_so['\gamma^{1}_{so}'] == 5), ['operator_label^{1}', 'operator_label^{0}']].values
        operator_so.loc[(operator_so['operator_label^{0}'] == '\gamma_{5}') & (operator_so['operator_label^{1}'] == '\gamma_{0}'), 'coefficient'] /= 2. 


    # Code doubled. May be refactored out.
    # Todo: Do I need to change the sign of q? I don't think so, because the daggering
    # taken care off when reading the data
    label = 'q_{si}'
    if label in operator_si.columns:

        operator_si.loc[operator_si[r'\gamma^{0}_{si}'].isin([0]), 'coefficient'] *= (-1)

        operator_si[label] = operator_si[label].apply(literal_eval)

        isospin_neg = operator_si[operator_si[label] <= (0, 0, 0)]
        isospin_neg.loc[:, 'coefficient'] *= -1
        isospin_neg.loc[:, label] = isospin_neg.loc[:, label].apply(utils._minus)

#        isospin_neg.loc[(isospin_neg['\gamma^{0}_{si}'] == isospin_neg['\gamma^{1}_{si}']), 'coefficient'] *= -1
#        isospin_neg.rename(columns={'\gamma^{0}_{si}': '\gamma^{1}_{si}',
#                                    '\gamma^{1}_{si}': '\gamma^{0}_{si}'}, inplace=True)
#        isospin_neg.rename(columns={'operator_label^{0}': 'operator_label^{1}',
#                                    'operator_label^{1}': 'operator_label^{0}'}, inplace=True)

        isospin_pos = operator_si[operator_si[label] >= (0, 0, 0)]
#        isospin_pos.loc[:, 'coefficient'] *= -1
#        isospin_pos.loc[isospin_pos['\gamma^{0}_{si}'] != isospin_pos['\gamma^{1}_{si}'], 'coefficient'] *= -1

#        isospin_pos.loc[(isospin_pos['\gamma^{0}_{si}'] == isospin_pos['\gamma^{1}_{si}']), 'coefficient'] *= -1
#        isospin_pos.rename(columns={'\gamma^{0}_{si}': '\gamma^{1}_{si}',
#                                    '\gamma^{1}_{si}': '\gamma^{0}_{si}'}, inplace=True)
#        isospin_pos.rename(columns={'operator_label^{0}': 'operator_label^{1}',
#                                    'operator_label^{1}': 'operator_label^{0}'}, inplace=True)

        operator_si = pd.concat([isospin_pos, isospin_neg])
        operator_si[label] = operator_si[label].apply(str)
        operator_si['coefficient'] /= np.sqrt(2)

        operator_si.loc[(operator_si['\gamma^{0}_{si}'] == 0) & (operator_si['\gamma^{1}_{si}'] == 5), ['operator_label^{0}', 'operator_label^{1}']] = operator_si.loc[(operator_si['\gamma^{0}_{si}'] == 0) & (operator_si['\gamma^{1}_{si}'] == 5), ['operator_label^{1}', 'operator_label^{0}']].values
        operator_si.loc[(operator_si['operator_label^{0}'] == '\gamma_{5}') & (operator_si['operator_label^{1}'] == '\gamma_{0}'), 'coefficient'] /= 2. 

    return operator_so, operator_si


def select_q(list_of_q, operators_so, operators_si):

    if list_of_q is not None:

        label = 'q_{so}'
        if label in operators_so.columns:
            operators_so.loc[:, label] = operators_so[label].apply(literal_eval)
            operators_so = operators_so[operators_so[label].isin(list_of_q)]
            operators_so.loc[:, label] = operators_so[label].apply(str)

        # Code doubled. May be refactored out.
        label = 'q_{si}'
        if label in operators_si.columns:
            operators_si.loc[:, label] = operators_si[label].apply(literal_eval)
            operators_si = operators_si[operators_si[label].isin(list_of_q)]
            operators_si.loc[:, label] = operators_si[label].apply(str)

    return operators_so, operators_si


def set_operator_labels(operators):

    operators['operator_label'] = operators[[
        col for col in operators.columns if 'label' in col]].apply(lambda x: ', '.join(x), axis=1)
    operators.drop(['operator_label^{0}', 'operator_label^{1}'],
                     axis=1, inplace=True, errors='ignore')

    return operators


def correlate_operators(operator_so, operator_si, verbose):

    # inner merge to get linear combinations of contributing correlation functions
    lattice_operators = pd.merge(operator_so, operator_si,
                                 how='inner', left_index=True, right_index=True,
                                 suffixes=['_{so}', '_{si}'])

    lattice_operators['coefficient'] = lattice_operators['coefficient_{so}'].apply(np.conj) \
        * lattice_operators['coefficient_{si}']
    lattice_operators.drop(
        ['coefficient_{so}', 'coefficient_{si}'], axis=1, inplace=True)

    lattice_operators.reset_index(inplace=True)
    index = lattice_operators.columns.difference(['coefficient']).tolist()
    order = {r'Irrep': 0,
             r'mult': 1,
             r'p_{cm}': 2,
             r'operator_label_{so}': 3,
             r'operator_label_{si}': 4,
             r'\mu': 5,
             r'\beta': 6,
             r'q_{so}': 7,
             r'q_{si}': 8,
             r'p^{0}_{so}': 9,
             r'p^{1}_{so}': 10,
             r'p^{0}_{si}': 11,
             r'p^{1}_{si}': 12,
             r'\gamma^{0}_{so}': 13,
             r'\gamma^{1}_{so}': 14,
             r'\gamma^{0}_{si}': 15,
             r'\gamma^{1}_{si}': 16}
    index = sorted(index, key=lambda x: order[x])
    lattice_operators.set_index(index, inplace=True)

    lattice_operators = lattice_operators.sum(axis=0, level=index)

    lattice_operators = lattice_operators[lattice_operators['coefficient'] != 0]

    if verbose >= 1:
        print 'lattice_operators'
    if verbose == 1:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print lattice_operators.head()
    if verbose >= 2:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print lattice_operators

    return lattice_operators


def project(j, correlator, continuum_basis_string, gamma_input, list_of_pcm, list_of_q,
            path_to_sc, path_to_sc_2, verbose):

    operators_so, operators_si = project_angular(j, correlator, continuum_basis_string, gamma_input, list_of_pcm, path_to_sc, path_to_sc_2, verbose)

    operators_so, operators_si = project_isospin(operators_so, operators_si)

    operators_so, operators_si = select_q(list_of_q, operators_so, operators_si)

    operators_so = set_operator_labels(operators_so)
    operators_si = set_operator_labels(operators_si)
    
    lattice_operators = correlate_operators(operators_so, operators_si, verbose)

    return lattice_operators
