import itertools as it
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import utils

def rho_2pt(p_cm, diagram='C20', verbose=0):

  # TODO: change the gamma structures
  gamma_i =   [1, 2, 3, 'gi']
  gamma_0i =  [10, 11, 12, 'g0gi']
  gamma_50i = [13, 14, 15, 'g5g0gi']
  
  gammas = [gamma_i, gamma_0i, gamma_50i]
  
  gamma_i = [(g,) for g in gamma_i[:-1]]
  gamma_0i = [(g,) for g in gamma_0i[:-1]]
  gamma_50i = [(g,) for g in gamma_50i[:-1]]


  data = pd.read_hdf('readdata/%s_p%1i.h5' % (diagram, p_cm), 'data')
  qn = pd.read_hdf('readdata/%s_p%1i.h5' % (diagram, p_cm), 'qn')

  wick = pd.concat([ \
  data.T[qn['\gamma_{so}'].isin(gamma_i)   & qn['\gamma_{si}'].isin(gamma_i)]  *( 2.), \
  data.T[qn['\gamma_{so}'].isin(gamma_i)   & qn['\gamma_{si}'].isin(gamma_0i)] *(-2.), \
  data.T[qn['\gamma_{so}'].isin(gamma_i)   & qn['\gamma_{si}'].isin(gamma_50i)]*( 2.*1j), \
  data.T[qn['\gamma_{so}'].isin(gamma_0i)  & qn['\gamma_{si}'].isin(gamma_i)]  *( 2.), \
  data.T[qn['\gamma_{so}'].isin(gamma_0i)  & qn['\gamma_{si}'].isin(gamma_0i)] *(-2.), \
  data.T[qn['\gamma_{so}'].isin(gamma_0i)  & qn['\gamma_{si}'].isin(gamma_50i)]*( 2.*1j), \
  data.T[qn['\gamma_{so}'].isin(gamma_50i) & qn['\gamma_{si}'].isin(gamma_i)]  *( 2.*1j), \
  data.T[qn['\gamma_{so}'].isin(gamma_50i) & qn['\gamma_{si}'].isin(gamma_0i)] *(-2.*1j), \
  data.T[qn['\gamma_{so}'].isin(gamma_50i) & qn['\gamma_{si}'].isin(gamma_50i)]*( 2.), \
  ]).sort_index().T

  # write data
  path = './readdata/%s_p%1i.h5' % ('C2', p_cm)
  utils.ensure_dir('./readdata')
  utils.write_hdf5_correlators(path, wick, qn)

################################################################################

def rho_3pt(p_cm, diagram='C3+', verbose=0):

  # TODO: change the gamma structures
  gamma_i =   [1, 2, 3, 'gi']
  gamma_0i =  [10, 11, 12, 'g0gi']
  gamma_50i = [13, 14, 15, 'g5g0gi']
  
  gammas = [gamma_i, gamma_0i, gamma_50i]
  
  gamma_i = [(g,) for g in gamma_i[:-1]]
  gamma_0i = [(g,) for g in gamma_0i[:-1]]
  gamma_50i = [(g,) for g in gamma_50i[:-1]]

  data = pd.read_hdf('readdata/%s_p%1i.h5' % (diagram, p_cm), 'data')
  qn = pd.read_hdf('readdata/%s_p%1i.h5' % (diagram, p_cm), 'qn')

  wick = pd.concat([ \
  data.T[qn['\gamma_{si}'].isin(gamma_i)]  *(2.), \
  data.T[qn['\gamma_{si}'].isin(gamma_0i)] *(-2.), \
  data.T[qn['\gamma_{si}'].isin(gamma_50i)]*(2*1j), \
  ]).sort_index().T

  # write data
  path = './readdata/%s_p%1i.h5' % ('C3', p_cm)
  utils.ensure_dir('./readdata')
  utils.write_hdf5_correlators(path, wick, qn)

################################################################################

def rho_4pt(p_cm, diagrams, verbose=0):

  data_box = pd.read_hdf('readdata/%s_p%1i.h5' % (diagrams[0], p_cm), 'data')
  qn_box = pd.read_hdf('readdata/%s_p%1i.h5' % (diagrams[0], p_cm), 'qn')
  data_dia = pd.read_hdf('readdata/%s_p%1i.h5' % (diagrams[1], p_cm), 'data')
  qn_dia = pd.read_hdf('readdata/%s_p%1i.h5' % (diagrams[1], p_cm), 'qn')


  # C4+D (data_dia) is the diagram calculated as product of two traces. To 
  # suppress noise, real and imaginary parts are seperated. Combine them 
  # according to (a+ib)*(c+id) = (ac-bd) +i(bc+ad)
  # to leave out noise contribution bd, just use
#                [data_dia.xs('rere', level=2), \
  # as real part.
  data_dia = pd.concat( \
                [data_dia.xs('rere', level=2) - data_dia.xs('imim', level=2), \
                 data_dia.xs('reim', level=2) - data_dia.xs('imre', level=2)], \
                                    keys=['re', 'im']).reorder_levels([1,2,0]).\
                                                         sort_index(level=[0,1])

  wick = ((-2.)*data_box).add(data_dia, fill_value=0)
  print wick[:3]

  # write data
  path = './readdata/%s_p%1i.h5' % ('C4', p_cm)
  utils.ensure_dir('./readdata')
  utils.write_hdf5_correlators(path, wick, qn_box)

################################################################################

# TODO: p_cm is only needed for filenames. Should be possible to go without it
def rho(p_cm, diagrams, verbose=0):
  """
  Sums all diagrams with the factors they appear in the Wick contractions

  Parameters
  ----------
  p_cm : int
      The center-of-mass momentum. 
  diagrams : list of string {'C20', 'C3+', 'C4+B', 'C4+D'}
      The diagrams the wick contration should be performed for
      WARNING: The order must be as above!

  Notes
  -----
  The correlation functions contributing to the rho gevp can be characterized
  by the number of quarklines 
  2pt \langle \rho(t_{so})^\dagger \rho(t_si) \rangle
  3pt \langle \pi\pi(t_{so})^\dagger \rho(t_si) \rangle
  4pt \langle \pi\pi(t_{so})^\dagger \pi\pi(t_si) \rangle
  In the isospin limit the first 2 only have one (linearly independent) diagram
  contributing, while the last one has two.

  The code currently expects to either give none or all diagrams contributing
  and is unsufficiently checking for erroneous calls.

  The alternative (hardcoding everything) was avoided, because now it is 
  possible to just read and contract a single gevp element.
  """

  # loop over number of quarklines and call the appropriate wick contraction
  # for each one
  # TODO: I don't think you have to emulate c function pointers for this
  # TODO: Refactor calculation of list correlators
  # TODO: make wick return correlators (data and qn)
  func_map = {2 : rho_2pt, 3 : rho_3pt, 4 : rho_4pt}
  correlators = []
  for nb_quarklines in range(2,5):
    # as a function argument give the names of all diagrams with the correct
    # number of quarklines
    mask = [d.startswith('C%1d' % nb_quarklines) for d in diagrams]
    diagram = list(it.compress(diagrams, mask))
    # compatibility to functions getting a single string, not a list
    # TODO make rho_2pt and rho_3pt accecpt a string so that this becomes 
    # superfluous
    if (len(diagram) == 0):
      continue
    correlators.append("C%1d" % nb_quarklines)
    if (len(diagram) == 1):
      diagram = diagram[0]
    if verbose:
      print diagram
    # call rho_2pt, rho_3pt, rho_4pt from this loop
    func_map[nb_quarklines](p_cm, diagram, verbose)

  return correlators

