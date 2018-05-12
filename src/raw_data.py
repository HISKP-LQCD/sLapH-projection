# functions for munging and cleaning correlators

import h5py
import numpy as np
import itertools as it

from utils import _minus
from raw_data_lookup_p import set_lookup_p
from raw_data_lookup_g import set_lookup_g

import functools

import pandas as pd
from pandas import Series, DataFrame

# TODO: nb_cnfg is spurious, can just use len(lookup_cnfg)
def set_lookup_cnfg(sta_cnfg, end_cnfg, del_cnfg, missing_configs, verbose=0):
  """
  Get a list of all gauge configurations contractions where performed on

  Parameters
  ----------
  sta_cnfg : int
      Number of first gauge configuration.
  end_cnfg : int
      Number of last gauge configuration.
  del_cnfg : int
      Step size chosen between following gauge configurations.
  missing_configs : list of int
      List of configurations to be omitted because the contractions were not
      performed

  Returns
  -------
  lookup_cnfg : list of int
      List of the configurations to read
  """

  # calculate number of configurations
  nb_cnfg = 0
  lookup_cnfg = []
  for cnfg in range(sta_cnfg, end_cnfg+1, del_cnfg):
    if cnfg in missing_configs:
      continue
    lookup_cnfg.append(cnfg)
  if(verbose >= 1):
    print '\t\tNumber of configurations: %i' % len(lookup_cnfg)

  return lookup_cnfg

def set_lookup_qn(diagram, p_cm, p_max, gammas, process='pipi', verbose=0):
  """
  Calculates a data frame with physical quantum numbers

  Parameters
  ----------
  diagram : string, {'C20', 'C2c', 'C3c', 'C4cB', 'C4cD'}
      Diagram of wick contractions for the rho meson.
  p_cm : int, {0, 1, 2, 3, 4}
      Center of mass momentum.
  p_max : int
      Maximum entry of momentum vectors.
  gammas : list of list of ints and string
      A list which for each gamma structure coupling to the rho meson contains
      a list with the integer indices used in the contraction code and a 
      latex-style name for plotting labels.

  Returns
  -------
  lookup_qn : pd.DataFrame
      pandas DataFrame where each row is a combination of quantum numbers and
      the row index is used as identifier for it.

  The quantum numbers are momenta and gamma structures at source and sink time.
  Depending on the number of quark lines there can be tuples of quantum numbers
  at the same lattice site
  """

  skip = True if process == 'rho' else False

  lookup_p = set_lookup_p(p_max, p_cm, diagram, skip)
  lookup_p.index = np.repeat(0, len(lookup_p))
  lookup_g = set_lookup_g(gammas, diagram)
  lookup_g.index = np.repeat(0, len(lookup_g))

  lookup_qn = pd.merge(lookup_p, lookup_g, how='left', left_index=True, right_index=True)
  lookup_qn.reset_index(drop=True, inplace=True)
  
  print lookup_qn
  return lookup_qn

# TODO: possible speedup if strings are directly used rather than the 
#       components of tuples
def set_groupname(diagram, s):
  """
  Creates string with filename of the desired diagram calculated by the 
  cntrv0.1-code.

  Parameters
  ----------
  diagram : string, {'C20', 'C2c', 'C3c', 'C4cB', 'C4cD'}
      Diagram of wick contractions for the rho meson.
  s : pd.Series
      Contains momenta and gamma structure the groupnames shall be built with.
      Index is given by Multiindex where the first level is 'p_{so}', 'p_{si}', 
      '\gamma_{so}' or '\gamma_{si}' and the second level id of the particle
      Dirac operators are referred to by their id in the sLapH-contractions
      code. 

  Returns
  -------
  groupname : string
      Filename of contracted perambulators for the given parameters

  Notes
  -----
  The filename contains a convential minus sign for the sink momenta!

  Function takes a series as argument in order to be called in DataFrame.apply()
  """

  if diagram.startswith('C2'):
    p_so = eval(s['p^{0}_{so}'])
    g_so = s['\gamma^{0}_{so}']
    p_si = _minus(eval(s['p^{0}_{si}']))
    g_si = s['\gamma^{0}_{si}']

    groupname = diagram \
                  + '_uu_p%1i%1i%1i.d000.g%i' % ( p_so + (g_so,) ) \
                  +    '_p%1i%1i%1i.d000.g%i' % ( p_si + (g_si,) ) 
  elif diagram.startswith('C3'):
    p_so_0 = eval(s['p^{0}_{so}'])
    p_so_1 = eval(s['p^{1}_{so}'])
    g_so_0 = s['\gamma^{0}_{so}']
    g_so_1 = s['\gamma^{1}_{so}']
    p_si = _minus(eval(s['p^{0}_{si}']))
    g_si = s['\gamma^{0}_{si}']

    groupname = diagram.replace('c', '+') \
                  + '_uuu_p%1i%1i%1i.d000.g%1i' % ( p_so_1 + (g_so_1,) ) \
                  +     '_p%1i%1i%1i.d000.g%1i' % ( p_si +   (g_si,) ) \
                  +     '_p%1i%1i%1i.d000.g%1i' % ( p_so_0 + (g_so_0,) )
  elif diagram == 'C4cD' or diagram == 'C4cC':
    p_so_0 = eval(s['p^{0}_{so}'])
    p_so_1 = eval(s['p^{1}_{so}'])
    g_so_0 = s['\gamma^{0}_{so}']
    g_so_1 = s['\gamma^{1}_{so}']
    p_si_0 = _minus(eval(s['p^{0}_{si}']))
    p_si_1 = _minus(eval(s['p^{1}_{si}']))
    g_si_0 = s['\gamma^{0}_{si}']
    g_si_1 = s['\gamma^{1}_{si}']

    groupname = diagram \
                  + '_uuuu_p%1i%1i%1i.d000.g%1i' % ( p_so_0 + (g_so_0,) ) \
                  +      '_p%1i%1i%1i.d000.g%1i' % ( p_si_0 + (g_si_0,) ) \
                  +      '_p%1i%1i%1i.d000.g%1i' % ( p_so_1 + (g_so_1,) ) \
                  +      '_p%1i%1i%1i.d000.g%1i' % ( p_si_1 + (g_si_1,) )
  elif diagram == 'C4cB':
    p_so_0 = eval(s['p^{0}_{so}'])
    p_so_1 = eval(s['p^{1}_{so}'])
    g_so_0 = s['\gamma^{0}_{so}']
    g_so_1 = s['\gamma^{1}_{so}']
    p_si_0 = _minus(eval(s['p^{0}_{si}']))
    p_si_1 = _minus(eval(s['p^{1}_{si}']))
    g_si_0 = s['\gamma^{0}_{si}']
    g_si_1 = s['\gamma^{1}_{si}']

    groupname = diagram \
                  + '_uuuu_p%1i%1i%1i.d000.g%1i' % ( p_so_0 + (g_so_0,) ) \
                  +      '_p%1i%1i%1i.d000.g%1i' % ( p_si_0 + (g_si_0,) ) \
                  +      '_p%1i%1i%1i.d000.g%1i' % ( p_si_1 + (g_si_1,) ) \
                  +      '_p%1i%1i%1i.d000.g%1i' % ( p_so_1 + (g_so_1,) ) 
  else:
    print 'in set_groupname: diagram unknown! Quantum numbers corrupted.'
    return

  return groupname

################################################################################
# reading configurations

def read(lookup_cnfg, lookup_qn, diagram, T, directory, verbose=0):
  """
  Read resulting correlators from contraction code and creates a pd.DataFrame

  Parameters
  ----------
  lookup_cnfg : list of int
      List of the gauge configurations to read
  lookup_qn : pd.DataFrame
      pd.DataFrame with every row being a set of physical quantum numbers to be 
      read
  diagram : string, {'C20', 'C2c', 'C3c', 'C4cB', 'C4cD'}
      Diagram of wick contractions for the rho meson.
  T : int
      Time extent of the lattice
  directory : string
      Output path of contraction code

  Returns
  -------
  data : pd.DataFrame
      A pd.DataFrame with rows (cnfg x T x re/im) and columns i where i are 
      the row numbers of `lookup_qn` 
  """

  comb = True if diagram == 'C4cD' else False

  groupname = lookup_qn.apply(functools.partial(set_groupname, diagram), axis=1)

  data = []

  for cnfg in lookup_cnfg:
    # filename and path
    filename = directory + '/' + diagram + '_cnfg%04i' % cnfg + '.h5'
    try:
      fh = h5py.File(filename, "r")
    except IOError:
      print 'file %s not found' % filename
      raise

    data_fh = DataFrame()

    for op in lookup_qn.index:

      # read data from file as numpy array and interpret as complex
      # numbers for easier treatment
      try:
        tmp = np.asarray(fh[groupname[op]]).view(complex)
      except KeyError:
        print("could not read %s for config %d" % (groupname, cnfg))
        continue
  
      # C4+D is the diagram factorizing into a product of two traces. The imaginary 
      # part of the individual traces is 0 in the isosping limit. To suppress noise, 
      # The real part of C4+D are calculated as product of real parts of the single 
      # traces:
      # (a+ib)*(c+id) ~= (ac) +i(bc+ad)
      # because b, d are noise
      if comb:
        # reshaping so we can extract the data easier
        tmp = tmp.reshape((-1,2))
        # extracting right combination, assuming ImIm contains only noise
        dtmp = 1.j * (tmp[:,1].real + tmp[:,0].imag) + tmp[:,0].real
        tmp = dtmp.copy()
  
      # save data into data frame
      data_fh[op] = pd.Series(tmp)
    data.append(data_fh)
  data = pd.concat(data, keys=lookup_cnfg, axis=0, names=['cnfg', 'T'])
  
 
  data.sort_index(level=[0,1], inplace=True)

  if verbose >= 1:
    print '\tfinished reading\n'
    print 'lookup_qn'
    print lookup_qn.head()
  if verbose >= 2:
    print 'lookup_qn'
    print lookup_qn
  if verbose >= 3:
    print 'data'
    print data.mean(axis=1).apply(np.real)

  return data

