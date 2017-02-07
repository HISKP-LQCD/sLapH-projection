import itertools as it
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import utils

def rho_2pt(data, irrep, verbose=0):

  # TODO: change the gamma structures
  gamma_i =   [1, 2, 3, 'gi']
  gamma_0i =  [10, 11, 12, 'g0gi']
  gamma_50i = [13, 14, 15, 'g5g0gi']
  
  gammas = [gamma_i, gamma_0i, gamma_50i]
  
  gamma_i = [(g,) for g in gamma_i[:-1]]
  gamma_0i = [(g,) for g in gamma_0i[:-1]]
  gamma_50i = [(g,) for g in gamma_50i[:-1]]


  wick = data[('C20',irrep)]

  idx = pd.IndexSclice

  wick.loc[idx[:,:,:,:,gamma_i,  :,gamma_i],  :] *= ( 2.)
  wick.loc[idx[:,:,:,:,gamma_i,  :,gamma_0i], :] *= (-2.)
  wick.loc[idx[:,:,:,:,gamma_i,  :,gamma_50i],:] *= ( 2.*1j)
  wick.loc[idx[:,:,:,:,gamma_0i, :,gamma_i],  :] *= ( 2.)
  wick.loc[idx[:,:,:,:,gamma_0i, :,gamma_0i], :] *= (-2.)
  wick.loc[idx[:,:,:,:,gamma_0i, :,gamma_50i],:] *= ( 2.*1j)
  wick.loc[idx[:,:,:,:,gamma_50i,:,gamma_i],  :] *= ( 2.*1j)
  wick.loc[idx[:,:,:,:,gamma_50i,:,gamma_0i], :] *= (-2.*1j)
  wick.loc[idx[:,:,:,:,gamma_50i,:,gamma_50i],:] *= ( 2.)

#  wick = pd.concat([ \
#  data.T[qn['\gamma_{so}'].isin(gamma_i)   & qn['\gamma_{si}'].isin(gamma_i)]  *( 2.), \
#  data.T[qn['\gamma_{so}'].isin(gamma_i)   & qn['\gamma_{si}'].isin(gamma_0i)] *(-2.), \
#  data.T[qn['\gamma_{so}'].isin(gamma_i)   & qn['\gamma_{si}'].isin(gamma_50i)]*( 2.*1j), \
#  data.T[qn['\gamma_{so}'].isin(gamma_0i)  & qn['\gamma_{si}'].isin(gamma_i)]  *( 2.), \
#  data.T[qn['\gamma_{so}'].isin(gamma_0i)  & qn['\gamma_{si}'].isin(gamma_0i)] *(-2.), \
#  data.T[qn['\gamma_{so}'].isin(gamma_0i)  & qn['\gamma_{si}'].isin(gamma_50i)]*( 2.*1j), \
#  data.T[qn['\gamma_{so}'].isin(gamma_50i) & qn['\gamma_{si}'].isin(gamma_i)]  *( 2.*1j), \
#  data.T[qn['\gamma_{so}'].isin(gamma_50i) & qn['\gamma_{si}'].isin(gamma_0i)] *(-2.*1j), \
#  data.T[qn['\gamma_{so}'].isin(gamma_50i) & qn['\gamma_{si}'].isin(gamma_50i)]*( 2.), \
#  ]).sort_index().T

  return wick

################################################################################

def rho_3pt(data, irrep, verbose=0):

  # TODO: change the gamma structures
  gamma_i =   [1, 2, 3, 'gi']
  gamma_0i =  [10, 11, 12, 'g0gi']
  gamma_50i = [13, 14, 15, 'g5g0gi']
  
  gammas = [gamma_i, gamma_0i, gamma_50i]
  
  gamma_i = [(g,) for g in gamma_i[:-1]]
  gamma_0i = [(g,) for g in gamma_0i[:-1]]
  gamma_50i = [(g,) for g in gamma_50i[:-1]]

  wick = data[('C3+',irrep)]

  wick.loc[idx[:,:,:,:,:,:,gamma_i],  :] *= ( 2.)
  wick.loc[idx[:,:,:,:,:,:,gamma_0i], :] *= (-2.)
  wick.loc[idx[:,:,:,:,:,:,gamma_50i],:] *= ( 2.*1j)

#  wick = pd.concat([ \
#  data.T[qn['\gamma_{si}'].isin(gamma_i)]  *(2.), \
#  data.T[qn['\gamma_{si}'].isin(gamma_0i)] *(-2.), \
#  data.T[qn['\gamma_{si}'].isin(gamma_50i)]*(2*1j), \
#  ]).sort_index().T

  return wick

################################################################################

# TODO: catch if keys were not found
# TODO: having to pass irrep is anoying, but the keys should vanish anyway when
# this is rewritten as member function
def rho_4pt(data, irrep, verbose=0):

  data_box = data[('C4+B', irrep)]
  data_dia = data[('C4+D', irrep)]
  
  # TODO: support read in if the passed data is incomplete
#  data_box = pd.read_hdf('readdata/%s_p%1i.h5' % (diagrams[0], p_cm), 'data')
#  data_dia = pd.read_hdf('readdata/%s_p%1i.h5' % (diagrams[1], p_cm), 'data')

  wick = ((-2.)*data_box).add(data_dia, fill_value=0)

  return wick

################################################################################

def set_lookup_correlators(diagrams):
  """
  Extracts the correlation functions one can construct from the given diagrams

  Parameters
  ----------
  diagrams : list of string {'C20', 'C3+', 'C4+B', 'C4+D'}
      Diagram of wick contractions contributing the rho meson correlation 
      function

  Returns
  -------
  lookup_correlators : list of string {'C2', 'C3', 'C4'}
      Correlation functions constituted by given diagrams
  """

  lookup_correlators = []
  for nb_quarklines in range(2,5):

    # as a function argument give the names of all diagrams with the correct
    # number of quarklines
    mask = [d.startswith('C%1d' % nb_quarklines) for d in diagrams]
    diagram = list(it.compress(diagrams, mask))

    if (len(diagram) == 0):
      continue
    else:
      lookup_correlators.append("C%1d" % nb_quarklines)

  return lookup_correlators

# TODO: pass path as alternative to read the data
def rho(data, correlator, irrep, verbose=0):
  """
  Sums all diagrams with the factors they appear in the Wick contractions

  Parameters
  ----------
  data : Dictionary of pd.DataFrame, keys in {'C20', 'C3+', 'C4+B', 'C4+d'}
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

