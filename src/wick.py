import itertools as it
import pandas as pd
from pandas import Series, DataFrame

import utils

def rho_2pt(data, irrep, verbose=1):
  """
  Perform Wick contraction for 2pt function

  Parameters
  ----------
  data : Dictionary of pd.DataFrame, keys in 
      ({'C20', 'C3c', 'C4cB', 'C4cD'}, `irrep`)

      For each diagram constituting the given `corrrelator` `data` must contain
      an associated pd.DataFrame with its subduced lattice data
  irrep : string, {'T1', 'A1', 'E2', 'B1', 'B2'}

      Name of the irreducible representation of the little group the operator
      is required to transform under.

  Returns
  -------
  wick : pd.DataFrame
      pd.DataFrame with indices like `data['C20']` and Wick contractions 
      performed.

  Notes
  -----
  The rho 2pt function is given by the contraction.
  C^\text{2pt} = \langle \rho(t_{so})^\dagger \rho(t_si) \rangle

  The gamma strucures that can appear in \rho(t) are hardcoded
  """

  gamma_i =   [1, 2, 3]
  gamma_50i = [13, 14, 15]

  wick = data[('C20',irrep)]

  # Warning: 1j hardcoded
  wick[wick.index.get_level_values('\gamma^{0}_{so}').isin(gamma_i) & \
       wick.index.get_level_values('\gamma^{0}_{si}').isin(gamma_i)]   *= (2.)
  wick[wick.index.get_level_values('\gamma^{0}_{so}').isin(gamma_i) & \
       wick.index.get_level_values('\gamma^{0}_{si}').isin(gamma_50i)] *= (2.*1j)
  wick[wick.index.get_level_values('\gamma^{0}_{so}').isin(gamma_50i) & \
       wick.index.get_level_values('\gamma^{0}_{si}').isin(gamma_i)]   *= (2.*1j)
  wick[wick.index.get_level_values('\gamma^{0}_{so}').isin(gamma_50i) & \
       wick.index.get_level_values('\gamma^{0}_{si}').isin(gamma_50i)] *= (2.)

  if verbose >= 2:
    print "wick[('C2', ", irrep, ")]"
  if verbose == 2:
    print wick.head()
  if verbose == 3:
    print wick

  return wick

def pipi_2pt(data, irrep, verbose=1):
  """
  Perform Wick contraction for 2pt function

  Parameters
  ----------
  data : Dictionary of pd.DataFrame, keys in 
      ({'C20', 'C3c', 'C4cB', 'C4cD'}, `irrep`)

      For each diagram constituting the given `corrrelator` `data` must contain
      an associated pd.DataFrame with its subduced lattice data
  irrep : string, {'T1', 'A1', 'E2', 'B1', 'B2'}

      Name of the irreducible representation of the little group the operator
      is required to transform under.

  Returns
  -------
  wick : pd.DataFrame
      pd.DataFrame with indices like `data['C20']` and Wick contractions 
      performed.

  Notes
  -----
  The gamma strucures that can appear in \pi\pi(t) are hardcoded
  """
  wick = data[('C2c',irrep)]

  return wick

################################################################################

def rho_3pt(data, irrep, verbose=1):
  """
  Perform Wick contraction for 3pt function

  Parameters
  ----------
  data : Dictionary of pd.DataFrame, keys in 
      ({'C20', 'C3c', 'C4cB', 'C4cD'}, `irrep`)

      For each diagram constituting the given `corrrelator` `data` must contain
      an associated pd.DataFrame with its subduced lattice data
  irrep : string, {'T1', 'A1', 'E2', 'B1', 'B2'}

      Name of the irreducible representation of the little group the operator
      is required to transform under.

  Returns
  -------
  wick : pd.DataFrame

      pd.DataFrame with indices like `data['C30']` and Wick contractions 
      performed.

  Notes
  -----
  The rho 3pt function is given by the contraction.
  C^\text{3pt} = \langle \pi\pi(t_{so})^\dagger \rho(t_si) \rangle

  The gamma strucures that can appear in \rho(t) are hardcoded
  """

  gamma_5 =   [5]
  gamma_i =   [1, 2, 3]
  gamma_50i = [13, 14, 15]

  wick = data[('C3c',irrep)]

  # Warning: 1j hardcoded
  wick[wick.index.get_level_values('\gamma^{0}_{so}').isin(gamma_5) & \
       wick.index.get_level_values('\gamma^{1}_{so}').isin(gamma_5) & \
       wick.index.get_level_values('\gamma^{0}_{si}').isin(gamma_i)]   *= ( 2.)   *(-1j)
  wick[wick.index.get_level_values('\gamma^{0}_{so}').isin(gamma_5) & \
       wick.index.get_level_values('\gamma^{1}_{so}').isin(gamma_5) & \
       wick.index.get_level_values('\gamma^{0}_{si}').isin(gamma_50i)] *= ( 2.*1j)*(-1j)

  if verbose >= 2:
    print "wick[('C3', ", irrep, ")]"
  if verbose == 2:
    print wick.head()
  if verbose == 3:
    print wick

  return wick

################################################################################

# TODO: catch if keys were not found
# TODO: having to pass irrep is anoying, but the keys should vanish anyway when
# this is rewritten as member function
def rho_4pt(data, irrep, verbose=0):
  """
  Perform Wick contraction for 4pt function

  Parameters
  ----------
  data : Dictionary of pd.DataFrame, keys in 
      ({'C20', 'C3c', 'C4cB', 'C4cD'}, `irrep`)

      For each diagram constituting the given `corrrelator` `data` must contain
      an associated pd.DataFrame with its subduced lattice data
  irrep : string, {'T1', 'A1', 'E2', 'B1', 'B2'}

      Name of the irreducible representation of the little group the operator
      is required to transform under.

  Returns
  -------
  wick : pd.DataFrame

      pd.DataFrame with indices like the union of indices in `data['C4cB']` and
      cdata['C4cD']` and Wick contractions performed.

  Notes
  -----
  The rho 3pt function is given by the contraction.
  C^\text{2pt} = \langle \pi\pi(t_{so})^\dagger \pi\pi(t_si) \rangle

  The gamma strucures that can appear in \rho(t) are hardcoded
  """

  data_box = data[('C4cB', irrep)]
  data_dia = data[('C4cD', irrep)]
  
  # TODO: support read in if the passed data is incomplete
#  data_box = pd.read_hdf('readdata/%s_p%1i.h5' % (diagrams[0], p_cm), 'data')
#  data_dia = pd.read_hdf('readdata/%s_p%1i.h5' % (diagrams[1], p_cm), 'data')

  wick = ((-2.)*data_box).add(data_dia, fill_value=0)

  if verbose >= 2:
    print "wick[('C4', ", irrep, ")]"
  if verbose == 2:
    print wick.head()
  if verbose == 3:
    print wick

  return wick

def pipi_4pt(data, irrep, verbose=0):
  """
  Perform Wick contraction for 4pt function

  Parameters
  ----------
  data : Dictionary of pd.DataFrame, keys in 
      ({'C20', 'C3c', 'C4cB', 'C4cD'}, `irrep`)

      For each diagram constituting the given `corrrelator` `data` must contain
      an associated pd.DataFrame with its subduced lattice data
  irrep : string, {'T1', 'A1', 'E2', 'B1', 'B2'}

      Name of the irreducible representation of the little group the operator
      is required to transform under.

  Returns
  -------
  wick : pd.DataFrame

      pd.DataFrame with indices like the union of indices in `data['C4cB']` and
      `data['C4cD']` and Wick contractions performed.

  Notes
  -----
  The rho 3pt function is given by the contraction.
  C^\text{2pt} = \langle \pi\pi(t_{so})^\dagger \pi\pi(t_si) \rangle

  The gamma strucures that can appear in \rho(t) are hardcoded
  """

  data_cro = data[('C4cC', irrep)]
  data_dia = data[('C4cD', irrep)]
  
  wick = ((-1.)*data_cro).add(data_dia, fill_value=0)

  return wick

################################################################################

def set_lookup_correlators(diagrams):
  """
  Extracts the correlation functions one can construct from the given diagrams

  Parameters
  ----------
  diagrams : list of string {'C20', 'C3c', 'C4cB', 'C4cD'}
      Diagram of wick contractions contributing the rho meson correlation 
      function

  Returns
  -------
  lookup_correlators : list of string {'C2', 'C3', 'C4'}
      Correlation functions constituted by given diagrams
  """

  lookup_correlators = {}
  for nb_quarklines in range(2,5):

    # as a function argument give the names of all diagrams with the correct
    # number of quarklines
    mask = [d.startswith('C%1d' % nb_quarklines) for d in diagrams]
    diagram = list(it.compress(diagrams, mask))

    if len(diagram) != 0:
        lookup_correlators.update({"C%1d" % nb_quarklines : diagram})

  return lookup_correlators

# TODO: pass path as alternative to read the data
def rho(data, correlator, irrep, verbose=0):
  """
  Sums all diagrams with the factors they appear in the Wick contractions

  Parameters
  ----------
  data : Dictionary of pd.DataFrame, keys in {'C20', 'C3c', 'C4cB', 'C4cD'}
      For each diagram constituting the given `corrrelator` `data` must contain
      an associated pd.DataFrame with its subduced lattice data
  correlator : string {'C2', 'C3', 'C4'}
      Correlation functions to perform the Wick contraction on
  irrep : string, {'T1', 'A1', 'E2', 'B1', 'B2'}
      name of the irreducible representation of the little group the operator
      is required to transform under.


  Returns
  -------
  contracted : pd.DataFrame
      A correlation function with completely performed Wick contractions. Rows
      and columns are unchanged compared to `data`

  Notes
  -----
  The correlation functions contributing to the rho gevp can be characterized
  by the number of quarklines 
  2pt \langle \rho(t_{so})^\dagger \rho(t_si) \rangle
  3pt \langle \pi\pi(t_{so})^\dagger \rho(t_si) \rangle
  4pt \langle \pi\pi(t_{so})^\dagger \pi\pi(t_si) \rangle
  In the isospin limit the first 2 only have one (linearly independent) diagram
  contributing, while the last one has two.
  """

  # TODO: I don't think you have to emulate c function pointers for this
  rho = {'C2' : rho_2pt, 'C3' : rho_3pt, 'C4' : rho_4pt}

  # call rho_2pt, rho_3pt, rho_4pt from this loop
  contracted = rho[correlator](data, irrep, verbose)

  return contracted

def pipi(data, correlator, irrep, verbose=0):
  """
  Sums all diagrams with the factors they appear in the Wick contractions

  Parameters
  ----------
  data : Dictionary of pd.DataFrame, keys in {'C20', 'C4cC', 'C4cD'}
      For each diagram constituting the given `corrrelator` `data` must contain
      an associated pd.DataFrame with its subduced lattice data
  correlator : string {'C2', 'C4'}
      Correlation functions to perform the Wick contraction on
  irrep : string, {'T1', 'A1', 'E2', 'B1', 'B2'}
      name of the irreducible representation of the little group the operator
      is required to transform under.


  Returns
  -------
  contracted : pd.DataFrame
      A correlation function with completely performed Wick contractions. Rows
      and columns are unchanged compared to `data`

  Notes
  -----
  The correlation functions contributing to the rho gevp can be characterized
  by the number of quarklines 
  2pt \langle \rho(t_{so})^\dagger \rho(t_si) \rangle
  3pt \langle \pi\pi(t_{so})^\dagger \rho(t_si) \rangle
  4pt \langle \pi\pi(t_{so})^\dagger \pi\pi(t_si) \rangle
  In the isospin limit the first 2 only have one (linearly independent) diagram
  contributing, while the last one has two.
  """

  # TODO: I don't think you have to emulate c function pointers for this
  pipi = {'C2' : pipi_2pt, 'C4' : pipi_4pt}

  # call rho_2pt, rho_3pt, rho_4pt from this loop
  contracted = pipi[correlator](data, irrep, verbose)

  return contracted
